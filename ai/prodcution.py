import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, LSTM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TimeSeriesForecasting") \
    .config("spark.jars", "postgresql-42.7.3.jar") \
    .getOrCreate()

logger.info("Spark session started")

# PostgreSQL connection properties
db_url = "jdbc:postgresql://localhost:5432/airflow"  # Replace with your PostgreSQL database
db_properties = {
    "user": "airflow",       # Replace with your PostgreSQL username
    "password": "airflow",   # Replace with your PostgreSQL password
    "driver": "org.postgresql.Driver"
}

logger.info("Connecting to PostgreSQL")

# Load the data from PostgreSQL table
table_name = "simplified_data"  # Replace with your table name
df = spark.read.jdbc(url=db_url, table=table_name, properties=db_properties)

logger.info("Data loaded from PostgreSQL")

df = df.withColumn('timestamp', col('timestamp').cast("long"))
df = df.orderBy('timestamp')
df = df.withColumn('price', col('price').cast(FloatType()))

logger.info("Data preprocessed")

# Define a UDF for sequence generation
sequence_length = 10

@pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
def generate_sequences(prices):
    prices_list = prices.tolist()
    sequences = []
    for i in range(len(prices_list) - sequence_length + 1):
        sequences.append(prices_list[i:i+sequence_length])
    # Padding for rows that do not have a full sequence
    for _ in range(sequence_length - 1):
        sequences.insert(0, [None] * sequence_length)
    return pd.Series(sequences)

# Apply the UDF to create sequences
df = df.withColumn('price_sequence', generate_sequences(col('price')))
df = df.dropna()

logger.info("Sequences generated and null values dropped")

# Split the data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

logger.info("Data split into training and testing sets")

# Define and compile the LSTM model
def build_model(input_shape):
    model = tf.keras.Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model

input_shape = (sequence_length, 1)
model = build_model(input_shape)

logger.info("LSTM model built and compiled")

# Train the model using the training data
@pandas_udf(FloatType(), PandasUDFType.SCALAR)
def train_model(price_sequence):
    logger.info("Starting model training for a batch")
    X = np.array(price_sequence.tolist()).reshape(-1, sequence_length, 1)
    y = X[:, -1, 0]
    
    try:
        model.fit(X, y, batch_size=32, epochs=50, )
        logger.info("Model training completed for a batch")
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise e
    
    # Return the last value of y for this segment
    return pd.Series([y[-1]] * len(price_sequence))

# Train the model with the training data
train_df = train_df.withColumn('training_output', train_model(col('price_sequence')))

logger.info("Model training completed for all batches")

# Predict using the trained model on the test data
@pandas_udf(FloatType(), PandasUDFType.SCALAR)
def predict_model(price_sequence):
    X = np.array(price_sequence.tolist()).reshape(-1, sequence_length, 1)
    
    try:
        predictions = model.predict(X).flatten()
        logger.info("Model prediction completed for a batch")
    except Exception as e:
        logger.error(f"Error during model prediction: {str(e)}")
        raise e
    
    return pd.Series(predictions)

# Apply the prediction UDF to the test data
test_df = test_df.withColumn('prediction', predict_model(col('price_sequence')))

logger.info("Model predictions completed for test data")

# Show the test data with predictions
test_df.show()

# Stop the Spark session
spark.stop()
logger.info("Spark session stopped")
