import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, from_unixtime

# Initialize Spark session
spark = SparkSession.builder.appName("LSTM_Model_Update").getOrCreate()

# Load data using Spark
file_path = '/mnt/data/last.csv'
data = spark.read.csv(file_path, header=True, inferSchema=True)

# Extract relevant columns and preprocess using Spark
data = data.drop('index')
data = data.withColumn('timestamp', from_unixtime(unix_timestamp('timestamp', 'yyyy-MM-dd HH:mm:ss')))
data = data.orderBy('timestamp')

# Convert Spark DataFrame to Pandas DataFrame for model training
data_pd = data.toPandas()
data_pd.set_index('timestamp', inplace=True)

# Preprocess data
def preprocess_data(data, sequence_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['price']].values)
    sequences = []
    if len(scaled_data) < sequence_length:
        print("Not enough data to create sequences")
        return np.array(sequences), scaler
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length + 1])
    return np.array(sequences), scaler


sequence_length = 50
train_size = int(len(data_pd) * 0.8)
train_data = data_pd.iloc[:train_size]
test_data = data_pd.iloc[train_size:]

train_sequences, train_scaler = preprocess_data(train_data, sequence_length)
test_sequences, _ = preprocess_data(test_data, sequence_length)

# Split into features and labels
if len(train_sequences) > 0 and len(test_sequences) > 0:
    X_train = train_sequences[:, :-1]
    y_train = train_sequences[:, -1]
    X_test = test_sequences[:, :-1]
    y_test = test_sequences[:, -1]

    # Build the RNN model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model initially
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save the latest timestamp after initial training
    latest_timestamp = data_pd.index[-1]
    save_latest_timestamp(latest_timestamp)

    # Continuous learning function
    def update_model_with_new_data(model, new_data_path, scaler, sequence_length, latest_timestamp):
        # Load new data using Spark
        new_data = spark.read.csv(new_data_path, header=True, inferSchema=True)
        new_data = new_data.withColumn('timestamp', from_unixtime(unix_timestamp('timestamp', 'yyyy-MM-dd HH:mm:ss')))
        new_data = new_data.orderBy('timestamp')
        
        # Filter new data
        new_data = new_data.filter(new_data['timestamp'] > latest_timestamp)
        
        # Convert Spark DataFrame to Pandas DataFrame
        new_data_pd = new_data.toPandas()
        new_data_pd.set_index('timestamp', inplace=True)
        
        new_sequences, _ = preprocess_data(new_data_pd, sequence_length)
        if len(new_sequences) > 0:
            X_new = new_sequences[:, :-1]
            y_new = new_sequences[:, -1]
            model.fit(X_new, y_new, epochs=1, batch_size=32)

            # Update the latest timestamp
            latest_timestamp = new_data_pd.index[-1]
            save_latest_timestamp(latest_timestamp)
        else:
            print("No new data available for updating the model.")

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Initial Test Loss: {loss}")

    # Example of updating the model with new data
    new_data_path = '/mnt/data/last.csv'
    latest_timestamp = load_latest_timestamp()
    update_model_with_new_data(model, new_data_path, train_scaler, sequence_length, latest_timestamp)

    # Re-evaluate the model after continuous learning
    loss = model.evaluate(X_test, y_test)
    print(f"Post-Update Test Loss: {loss}")
else:
    print("Not enough sequences generated. Please check the data or sequence length.")
