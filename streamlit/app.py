import streamlit as st
import requests
import pandas as pd

# Streamlit App
st.title("Time Series Prediction App")
st.write("Enter feature values to predict the target variable.")

# Define input features
def user_input_features():
    timestamp = st.number_input("Timestamp", min_value=0)
    lastupdateid = st.number_input("Last Update ID", min_value=0)
    pricechange = st.number_input("Price Change", value=0.0)
    pricechangepercent = st.number_input("Price Change Percent", value=0.0)
    weightedavgprice = st.number_input("Weighted Average Price", value=0.0)
    prevcloseprice = st.number_input("Previous Close Price", value=0.0)
    lastprice = st.number_input("Last Price", value=0.0)
    lastqty = st.number_input("Last Quantity", value=0.0)
    bidprice = st.number_input("Bid Price", value=0.0)
    bidqty = st.number_input("Bid Quantity", value=0.0)
    askprice = st.number_input("Ask Price", value=0.0)
    askqty = st.number_input("Ask Quantity", value=0.0)
    openprice = st.number_input("Open Price", value=0.0)
    highprice = st.number_input("High Price", value=0.0)
    lowprice = st.number_input("Low Price", value=0.0)
    volume = st.number_input("Volume", value=0.0)
    quotevolume = st.number_input("Quote Volume", value=0.0)

    data = {
        'timestamp': timestamp,
        'lastupdateid': lastupdateid,
        'pricechange': pricechange,
        'pricechangepercent': pricechangepercent,
        'weightedavgprice': weightedavgprice,
        'prevcloseprice': prevcloseprice,
        'lastprice': lastprice,
        'lastqty': lastqty,
        'bidprice': bidprice,
        'bidqty': bidqty,
        'askprice': askprice,
        'askqty': askqty,
        'openprice': openprice,
        'highprice': highprice,
        'lowprice': lowprice,
        'volume': volume,
        'quotevolume': quotevolume
    }

    return data

input_data = user_input_features()

# Prediction button
if st.button("Predict"):
    # Send the data to the FastAPI server
    try:
        response = requests.post("http://localhost:2000/predict/", json=input_data)
        response_data = response.json()
        predicted_value = response_data['prediction']

        st.write(f"Predicted Price: ${predicted_value:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Display input features
st.write("Input Features")
st.write(pd.DataFrame([input_data]))