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
    'binance_btcusdt_klines_dag',
    default_args=default_args,
    description='Fetch BTCUSDT Kline data from Binance',
    schedule_interval=timedelta(minutes=10),
    start_date=days_ago(1),
    catchup=False,
)

def fetch_klines():
    url = 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m'
    data = fetch_data_sync(url)
    # Process data here
    print(data)

fetch_klines_task = PythonOperator(
    task_id='fetch_klines',
    python_callable=fetch_klines,
    dag=dag,
)

fetch_klines_task
