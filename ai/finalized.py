# Import required libraries
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Specify the path to the PostgreSQL JDBC driver
jdbc_driver_path = "postgresql-42.7.3.jar"  # Replace with the actual path to the JDBC driver

# Initialize Spark session with the JDBC driver
spark = SparkSession.builder \
    .master("local") \
    .appName("TimeSeriesForecasting") \
    .config("spark.jars", jdbc_driver_path) \
    .getOrCreate()

# PostgreSQL connection properties
db_url = "jdbc:postgresql://localhost:5432/airflow"  # Replace with your actual PostgreSQL URL
db_properties = {
    "user": "airflow",       # Replace with your PostgreSQL username
    "password": "airflow",   # Replace with your PostgreSQL password
    "driver": "org.postgresql.Driver"
}

# Load the data from PostgreSQL table
table_name = "simplified_data"  # Replace with your table name
df = spark.read.jdbc(url=db_url, table=table_name, properties=db_properties)

# Show the first few rows of the dataframe
df.show(5)

# Preprocess the data
# Replace 'timestamp_column' and 'target_column' with the actual column names
feature_columns = [col for col in df.columns if col != 'timestamp' and col != 'price']
target_column = 'price'

# Select the feature columns and target column
df = df.select(*feature_columns, target_column).orderBy("timestamp")

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = df.toPandas().dropna()

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(pandas_df)

# Separate features and target
X = scaled_data[:, :-1]  # All columns except the last one (which is the target)
Y = scaled_data[:, -1]   # The last column is the target (price)

# Reshape input to be [samples, time steps, features] for LSTM
# Using a time_step for sequence data
def create_dataset(X, Y, time_step=1):
    Xs, Ys = [], []
    for i in range(len(X) - time_step - 1):
        Xs.append(X[i:(i + time_step), :])
        Ys.append(Y[i + time_step])
    return np.array(Xs), np.array(Ys)

time_step = 10
X, Y = create_dataset(X, Y, time_step)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))  # Use X.shape[2] for the number of features
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error')

def check_nan_in_weights(epoch, logs):
    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            if np.any(np.isnan(w)):
                print(f"NaN detected in layer {layer.name} weights during epoch {epoch}")
                break

nan_checker = tf.keras.callbacks.LambdaCallback(on_epoch_end=check_nan_in_weights)

# Train the model
model.fit(X, Y, batch_size=1, epochs=10, callbacks=[nan_checker])

# Make predictions
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

# Visualize the results
plt.plot(pandas_df['timestamp_column'], scaler.inverse_transform(scaled_data), label='True Data')
plt.plot(pandas_df['timestamp_column'][time_step:], predictions, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Forecasting')
plt.legend()
plt.show()
