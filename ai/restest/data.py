import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TimeSeriesForecasting") \
    .config("spark.jars", "postgresql-42.7.3.jar") \
    .getOrCreate()

def load_and_preprocess_data():
    # PostgreSQL connection properties
    db_url = "jdbc:postgresql://localhost:5432/airflow"  # Replace with your PostgreSQL database
    db_properties = {
        "user": "airflow",
        "password": "airflow",
        "driver": "org.postgresql.Driver"
    }

    # Load the data from PostgreSQL table
    table_name = "simplified_data"  # Replace with your table name
    df = spark.read.jdbc(url=db_url, table=table_name, properties=db_properties)
    df = df.withColumn('timestamp', col('timestamp').cast("long"))
    df = df.orderBy('timestamp')

    # Preprocess data
    for col_name in df.columns:
        if col_name != "timestamp":
            df = df.withColumn(col_name, col(col_name).cast(FloatType()))

    pandas_df = df.toPandas()
    pandas_df.dropna(inplace=True)
    
    return pandas_df

def generate_sequences(pandas_df, scaler_X, scaler_y):
    # Separate input features and target (price)
    X_columns = pandas_df.columns.difference(['price'])
    y_column = 'price'

    # Normalize data
    pandas_df[X_columns] = scaler_X.fit_transform(pandas_df[X_columns])
    pandas_df[y_column] = scaler_y.fit_transform(pandas_df[[y_column]])

    sequence_length = 10
    X_sequences = []
    y_values = []

    for i in range(len(pandas_df) - sequence_length):
        X_sequence = pandas_df[X_columns].iloc[i:i + sequence_length].values.astype(np.float32)
        y_value = pandas_df[y_column].iloc[i + sequence_length].astype(np.float32)
        
        X_sequences.append(X_sequence)
        y_values.append(y_value)

    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_values = np.array(y_values, dtype=np.float32)

    return X_sequences, y_values

def clear_simplified_data_table():
    try:
        # db_url = "jdbc:postgresql://localhost:5432/airflow"  # Replace with your PostgreSQL database URL
        # db_properties = {
        #     "user": "airflow",
        #     "password": "airflow",
        #     "driver": "org.postgresql.Driver"
        # }
        
        # # Execute TRUNCATE TABLE command to clear the `simplified_data` table
        # spark.sql(f"TRUNCATE TABLE {db_properties['user']}.simplified_data")
        print("simplified_data table cleared successfully.")
    except Exception as e:
        print(f"Error clearing simplified_data table: {str(e)}")
