import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Load data
file_path = 'test.csv'
data = pd.read_csv(file_path)

# Extract relevant columns
data.drop(columns=['index'], inplace=True)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.sort_values('timestamp')
data.set_index('timestamp', inplace=True)

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

sequence_length = 25
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

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

    # Continuous learning function
    def update_model_with_new_data(model, new_data_path, scaler, sequence_length):
        new_data = pd.read_csv(new_data_path)
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        new_data = new_data.sort_values('timestamp')
        new_data.set_index('timestamp', inplace=True)
        
        # Find the new data (assuming new data is appended after the initial data)
        new_data = new_data[~new_data.index.isin(data.index)]
        
        new_sequences, _ = preprocess_data(new_data, sequence_length)
        if len(new_sequences) > 0:
            X_new = new_sequences[:, :-1]
            y_new = new_sequences[:, -1]
            model.fit(X_new, y_new, epochs=1, batch_size=32)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Initial Test Loss: {loss}")

    # Example of updating the model with new data
    new_data_path = 'test.csv'
    update_model_with_new_data(model, new_data_path, train_scaler, sequence_length)

    # Re-evaluate the model after continuous learning
    loss = model.evaluate(X_test, y_test)
    print(f"Post-Update Test Loss: {loss}")
else:
    print("Not enough sequences generated. Please check the data or sequence length.")
