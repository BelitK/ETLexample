from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from utils import fetch_data_sync

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'binance_btcusdt_avgprice_dag',
    default_args=default_args,
    description='Fetch BTCUSDT average price from Binance',
    schedule_interval=timedelta(minutes=10),
    start_date=days_ago(1),
    catchup=False,
)

def fetch_avgprice():
    url = 'https://api.binance.com/api/v3/avgPrice?symbol=BTCUSDT'
    data = fetch_data_sync(url)
    # Process data here
    print(data)

fetch_avgprice_task = PythonOperator(
    task_id='fetch_avgprice',
    python_callable=fetch_avgprice,
    dag=dag,
)

fetch_avgprice_task
