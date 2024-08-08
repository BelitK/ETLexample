from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta, datetime
import logging
import pandas as pd
from utils import get_data, insert_data, update_data


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'transform_btcusdt_data_dag',
    default_args=default_args,
    description='Extract, transform, and load BTCUSDT data',
    schedule_interval=timedelta(minutes=10),
    start_date=days_ago(1),
    catchup=False,
)

def extract_data():
    try:
        price_data = get_data('crypto_price_data')
        ticker_data = get_data('crypto_ticker')
        depth_data = get_data('crypto_depth')
        return price_data, ticker_data, depth_data
    except Exception as e:
        raise RuntimeError(f"Error : {e}")

def transform_data(ti):
    price_data, ticker_data, depth_data = ti.xcom_pull(task_ids='extract_data')
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

    combined_data = merged_df.drop(columns=['opentime','closetime','firstid','lastid','count','bids','asks'])
    return combined_data

def load_data(ti):
    combined_data = ti.xcom_pull(task_ids='transform_data')
    try:
        insert_data('simplified_data',combined_data, conn_type='alch')
    except Exception as e:
        raise RuntimeError(f"Error loading data into the database: {e}")
    
def update_source(ti):
    price_data, ticker_data, depth_data = ti.xcom_pull(task_ids='extract_data')
    
    try:
        
        for key, value in zip([price_data, ticker_data, depth_data], ['crypto_price_data','crypto_ticker', 'crypto_depth']):
            update_data(value, key['id'].to_list())
    except Exception as e:
        raise RuntimeError(f"Error loading data into the database: {e}")

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

update_source = PythonOperator(
    task_id='update_souce',
    python_callable=update_source,
    provide_context=True,
    dag=dag,
)

extract_task >> transform_task >> load_task >> update_source
