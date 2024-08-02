from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import requests
from utils import fetch_data_sync, insert_data


def extract(**kwargs):
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        data = fetch_data_sync(url)
        kwargs["ti"].xcom_push(key="raw_data", value=data)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching data from Binance: {e}")


def transform(**kwargs):
    try:
        ti = kwargs["ti"]
        data = ti.xcom_pull(key="raw_data", task_ids="extract_btcusdt_data") # gotta use function names
        if "price" not in data:
            raise ValueError("Price data not found in the fetched data")
        data["price"] = str(round(float(data["price"]), 2))
        ti.xcom_push(key="transformed_data", value=data)
    except Exception as e:
        raise RuntimeError(f"Error transforming data: {e}")


def load(**kwargs):
    try:
        ti = kwargs["ti"]
        data = ti.xcom_pull(key="transformed_data", task_ids="transform_btcusdt_data")
        insert_data("crypto_price_data", data)
    except Exception as e:
        raise RuntimeError(f"Error connecting to the database: {e}")


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "binance_btcusdt_etl_dag",
    default_args=default_args,
    description="A simple ETL DAG to fetch, transform, and store BTCUSDT data from Binance",
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
)

start_task = DummyOperator(task_id="start", dag=dag)

extract_task = PythonOperator(
    task_id="extract_btcusdt_data",
    python_callable=extract,
    provide_context=True,
    dag=dag,
)

transform_task = PythonOperator(
    task_id="transform_btcusdt_data",
    python_callable=transform,
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

start_task >> extract_task >> transform_task >> load_task >> end_task
