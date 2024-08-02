from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import requests
from utils import fetch_data_sync, insert_data

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'binance_btcusdt_ticker_dag',
    default_args=default_args,
    description='Fetch BTCUSDT 24-hour ticker data from Binance',
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
)

def fetch_ticker(**kwargs):
    try:
        url = url = 'https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT'
        data = fetch_data_sync(url)
        kwargs["ti"].xcom_push(key="raw_data", value=data)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching data from Binance: {e}")

def load(**kwargs):
    try:
        ti = kwargs["ti"]
        data = ti.xcom_pull(key="raw_data", task_ids="fetch_ticker")
        insert_data("crypto_ticker", data)
    except Exception as e:
        raise RuntimeError(f"Error connecting to the database: {e}")


start_task = DummyOperator(task_id="start", dag=dag)

extract_task = PythonOperator(
    task_id="fetch_ticker",
    python_callable=fetch_ticker,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
    task_id="load_btcusdt_data",
    python_callable=load,
    provide_context=True,
    dag=dag,
)

end_task = DummyOperator(task_id="end", dag=dag)

start_task >> extract_task >> load_task >> end_task
