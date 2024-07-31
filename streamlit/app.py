import streamlit as st
import pandas as pd
import time
from data_realtime import fetch_data_from_db, generate_conn


# Define the Streamlit app
st.title('Real-Time Dashboard')

# Dropdown to select data table
table = st.selectbox('Select Data Table', ['btcusdt_data', 'solusdt_data'], index=0)

# Placeholder for data display
data_placeholder = st.empty()

# Fetch and display data in real-time
while True:
    data = fetch_data_from_db(conn=generate_conn(), table_name=table)
    data_placeholder.write(data)
    
    time.sleep(5)  # Refresh every 5 seconds
