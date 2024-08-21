import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import time

# Paths for saving the models and scalers
model_dir = "models"
temp_model_path = os.path.join(model_dir, "temp_model.keras")
scaler_X_path = "scaler_X.pkl"
scaler_y_path = "scaler_y.pkl"
current_model_path = None  # Use this variable to track the current model

@tf.keras.utils.register_keras_serializable()
def custom_mse():
    return tf.keras.losses.MeanSquaredError()


def build_model(input_shape):
    model = tf.keras.Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='relu')  # ReLU activation ensures non-negative outputs
    ])
    optimizer = Adam(learning_rate=0.001, clipvalue=1.0)

    # Use explicit references to loss function and metrics
    loss_fn = custom_mse()
    metrics = [tf.keras.metrics.MeanSquaredError()]

    print(f"Compiling model with loss: {loss_fn}, metrics: {metrics}")

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return model


def load_model_and_scalers(temp=False):
    global current_model_path  # Ensure we update the global variable

    try:
        model = None

        # Attempt to load the temporary model if specified
        if temp and os.path.exists(temp_model_path):
            print(f"Loading temporary model from {temp_model_path}")
            model = tf.keras.models.load_model(temp_model_path, custom_objects={'custom_mse': custom_mse})
            print(f"Temporary model loaded from {temp_model_path}.")
        # Attempt to load the current model using the current_model_path variable
        elif current_model_path and os.path.exists(current_model_path):
            model = tf.keras.models.load_model(current_model_path, custom_objects={'custom_mse': custom_mse})
            print(f"Model loaded from {current_model_path}.")
        # If no current model, attempt to find the most recent model
        else:
            versions = sorted(os.listdir(model_dir), reverse=True)
            if versions:
                latest_version = versions[0]
                latest_model_path = os.path.join(model_dir, latest_version, f"model_{latest_version}.keras")
                model = tf.keras.models.load_model(latest_model_path, custom_objects={'custom_mse': custom_mse})
                current_model_path = latest_model_path
                print(f"Model loaded from {latest_model_path}.")

        # If no models are found, initialize a new model
        if model is None:
            model = build_model((10, 17))  # Placeholder input shape: (timesteps, features)
            print("New model initialized.")

        print(f"Model loaded successfully. Loss: {model.loss}, Metrics: {model.metrics_names}")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        print("Scalers loaded from disk.")
    else:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        print("Scalers initialized.")

    return model, scaler_X, scaler_y


def save_temp_model(model):
    try:
        print(f"Saving temporary model to {temp_model_path}")
        model.save(temp_model_path, save_format='keras')
        print("Temporary model saved to disk.")
    except Exception as e:
        print(f"Error saving temporary model: {e}")
        raise e


def save_model_and_scalers(model, scaler_X, scaler_y, validated=False):
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        version_dir = os.path.join(model_dir, timestamp)
        os.makedirs(version_dir, exist_ok=True)

        model_path = os.path.join(version_dir, f"model_{timestamp}.keras")
        print(f"Saving model to {model_path}")
        model.save(model_path, save_format='keras')

        joblib.dump(scaler_X, os.path.join(version_dir, scaler_X_path))
        joblib.dump(scaler_y, os.path.join(version_dir, scaler_y_path))

        print(f"Model and scalers saved to {version_dir}.")

        if validated:
            current_model_symlink = model_path
            # Replace or create the symlink
            print(f"Symlink updated to new model version: {timestamp}")

    except Exception as e:
        print(f"Error saving model or scalers: {e}")
        raise e


def rollback_to_previous_model():
    try:
        versions = sorted(os.listdir(model_dir), reverse=True)
        if len(versions) < 2:
            print("No previous model to rollback to.")
            return

        previous_version = versions[1]
        previous_model_path = os.path.join(model_dir, previous_version, f"model_{previous_version}.keras")

        current_model_symlink = previous_model_path
        print(f"Rolled back to previous model version: {previous_version}")

    except Exception as e:
        print(f"Error during rollback: {e}")
        raise e