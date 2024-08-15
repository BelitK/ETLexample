import streamlit as st
import requests

# Set the base URL for your FastAPI application
BASE_URL = "http://localhost:8000"  # Update with your FastAPI app's URL if different

# Page title
st.title("Continuous Learning Model Dashboard")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Train Model", "Validate Model", "Continuous Learning", "Rollback"])

# Prediction Page
if page == "Predict":
    st.header("Make a Prediction")
    
    # Input features
    st.subheader("Enter Feature Values")
    feature1 = st.number_input("Feature 1")
    feature2 = st.number_input("Feature 2")
    feature3 = st.number_input("Feature 3")
    # Add more feature inputs as needed
    
    features = [feature1, feature2, feature3]  # Adjust based on the number of features
    
    if st.button("Predict"):
        # Make a prediction request
        response = requests.post(f"{BASE_URL}/predict", json={"features": features})
        
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Error: {response.json()['detail']}")

# Train Model Page
elif page == "Train Model":
    st.header("Train the Model")
    
    if st.button("Start Training"):
        response = requests.post(f"{BASE_URL}/train")
        
        if response.status_code == 200:
            st.success("Training started in the background.")
        else:
            st.error(f"Error: {response.json()['detail']}")

# Validate Model Page
elif page == "Validate Model":
    st.header("Validate and Deploy the Model")
    
    if st.button("Start Validation"):
        response = requests.post(f"{BASE_URL}/validate-model")
        
        if response.status_code == 200:
            st.success("Validation and deployment started in the background.")
        else:
            st.error(f"Error: {response.json()['detail']}")

# Continuous Learning Page
elif page == "Continuous Learning":
    st.header("Start Continuous Learning")
    
    if st.button("Start Continuous Learning"):
        response = requests.post(f"{BASE_URL}/continuous-learning")
        
        if response.status_code == 200:
            st.success("Continuous learning started in the background.")
        else:
            st.error(f"Error: {response.json()['detail']}")

# Rollback Page
elif page == "Rollback":
    st.header("Rollback to Previous Model")
    
    if st.button("Rollback"):
        response = requests.post(f"{BASE_URL}/rollback")
        
        if response.status_code == 200:
            st.success("Rollback to the previous model version completed.")
        else:
            st.error(f"Error: {response.json()['detail']}")
