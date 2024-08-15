from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from model import build_model, load_model_and_scalers, save_temp_model, save_model_and_scalers, rollback_to_previous_model
from data import load_and_preprocess_data, generate_sequences, clear_simplified_data_table

app = FastAPI()

# Global variable to store performance metrics for rollback decision
performance_metrics = {"mse": float('inf')}

# Request model for prediction input
class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    model, scaler_X, scaler_y = load_model_and_scalers()
    
    try:
        input_data = np.array(request.features).reshape(1, -1)
        scaled_input = scaler_X.transform(input_data).reshape(1, -1, len(input_data[0]))
        
        # Reshape input for LSTM: [batch_size, timesteps, features]
        timesteps = 1  # Since we're predicting with a single instance
        features = 17  # Number of features
        scaled_input = scaled_input.reshape((1, timesteps, features))
        
        scaled_prediction = model.predict(scaled_input)
        prediction = scaler_y.inverse_transform(scaled_prediction).flatten()
        
        # Update performance metrics (e.g., MSE)
        update_performance_metrics(scaled_input, prediction)
        
        # Automatically check if rollback is needed
        check_rollback_criteria()
        
        return {"prediction": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def update_performance_metrics(input_data, prediction):
    # Example: Calculate MSE using dummy actual values; in real cases, use actual values
    actual_values = np.array(input_data).flatten()
    prediction = np.array(prediction).flatten()
    mse = mean_squared_error(actual_values, prediction)
    
    # Update the global performance metrics
    performance_metrics["mse"] = mse

def check_rollback_criteria():
    mse_threshold = 2000  # Define a threshold for MSE that triggers rollback
    
    if performance_metrics["mse"] > mse_threshold:
        print("Performance threshold breached. Rolling back to the previous model.")
        rollback_to_previous_model()

@app.post("/train")
def train_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_temp_model_in_background)
    return {"message": "Temporary training started in the background."}

def train_temp_model_in_background():
    model, scaler_X, scaler_y = load_model_and_scalers(temp=True)
    
    pandas_df = load_and_preprocess_data()

    # Split the data into training and validation sets
    train_df = pandas_df.sample(frac=0.8, random_state=42)  # 80% training data
    val_df = pandas_df.drop(train_df.index)  # 20% validation data
    
    # Generate sequences for training and validation
    X_train, y_train = generate_sequences(train_df, scaler_X, scaler_y)
    X_val, y_val = generate_sequences(val_df, scaler_X, scaler_y)
    
    # Debugging: Print the original shapes
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original y_train shape: {y_train.shape}")

    timesteps = 10  # Sequence length
    features = 17  # Number of features

    # Reshape input for LSTM: [batch_size, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], timesteps, features))
    X_val = X_val.reshape((X_val.shape[0], timesteps, features))

    batch_size = 16
    epochs_per_batch = 5
    num_batches = len(X_train) // batch_size

    for batch_num in range(num_batches):
        X_batch = X_train[batch_num * batch_size:(batch_num + 1) * batch_size]
        y_batch = y_train[batch_num * batch_size:(batch_num + 1) * batch_size]
        
        model.fit(X_batch, y_batch, batch_size=batch_size, epochs=epochs_per_batch, verbose=1)
    
    # Save the model as a temporary model
    save_temp_model(model)

    # Evaluate the model on the validation set
    val_predictions = model.predict(X_val).flatten()
    val_predictions = scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
    y_val = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

    # Calculate validation metrics
    mse = mean_squared_error(y_val, val_predictions)
    print(f"Validation MSE: {mse}")

    # Save the MSE for validation decision
    return mse

@app.post("/validate-model")
def validate_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(validate_and_deploy_model)
    return {"message": "Validation and deployment started in the background."}

def validate_and_deploy_model():
    mse = train_temp_model_in_background()

    # Set a threshold for validation
    validation_threshold = 100000  # Adjust this threshold based on your problem

    if mse < validation_threshold:
        model, scaler_X, scaler_y = load_model_and_scalers(temp=True)
        save_model_and_scalers(model, scaler_X, scaler_y, validated=True)
        print("New model deployed.")
        
        # Clear the `simplified_data` table after successful validation
        clear_simplified_data_table()
        
    else:
        print("Model validation failed. Keeping the previous model.")

@app.post("/rollback")
def rollback():
    rollback_to_previous_model()
    return {"message": "Rolled back to the previous model version."}

@app.post("/continuous-learning")
def continuous_learning(background_tasks: BackgroundTasks):
    background_tasks.add_task(continuous_learning_loop)
    return {"message": "Continuous learning started in the background."}

def continuous_learning_loop():
    while True:
        mse = train_temp_model_in_background()

        if mse < 100000:  # Replace 1000 with your validation threshold
            validate_and_deploy_model()

        # Wait for the next learning cycle (e.g., every 1 hour)
        time.sleep(3600)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
