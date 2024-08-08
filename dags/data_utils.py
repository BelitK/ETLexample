import pandas as pd
from sqlalchemy import create_engine

# Define the PostgreSQL URL
postgresql_url = 'postgresql://airflow:airflow@173.212.221.185:5432/airflow'

# Create an engine
engine = create_engine(postgresql_url)

# Establish a connection
connection = engine.connect()

price_data = pd.read_sql('Select * from crypto_price_data where processed = False;',connection)
depth_data = pd.read_sql('Select * from crypto_depth where processed = False;',connection)
ticker_data = pd.read_sql('Select * from crypto_ticker where processed = False;',connection)

df1 = price_data.drop(columns=['id','symbol','processed'])
df2 = depth_data.drop(columns=['id','processed'])
df3 = ticker_data.drop(columns=['id','symbol','processed'])

# Convert 'timestamp' column to datetime
df1['timestamp'] = pd.to_datetime(df1['timestamp'])
df2['timestamp'] = pd.to_datetime(df2['timestamp'])
df3['timestamp'] = pd.to_datetime(df3['timestamp'])

# Sort DataFrames by 'timestamp'
df1.sort_values('timestamp', inplace=True)
df2.sort_values('timestamp', inplace=True)
df3.sort_values('timestamp', inplace=True)

# Merge with tolerance
tolerance = pd.Timedelta('4min')

merged_df = pd.merge_asof(df1, df2, on='timestamp', tolerance=tolerance, direction='nearest')
merged_df = pd.merge_asof(merged_df, df3, on='timestamp', tolerance=tolerance, direction='nearest')

merged_df.drop(columns=['opentime','closetime','firstid','lastid','count','bids','asks'], inplace=True)

