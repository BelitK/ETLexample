from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import requests
import json
import psycopg2

def extract(**kwargs):
    try:
        url = 'https://api.binance.com/api/v3/ticker/price?symbol=SOLUSDT'
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError if the HTTP request returned an unsuccessful status code
        data = response.json()
        kwargs['ti'].xcom_push(key='raw_data', value=data)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error fetching data from Binance: {e}")

def transform(**kwargs):
    try:
        ti = kwargs['ti']
        data = ti.xcom_pull(key='raw_data', task_ids='extract_btcusdt_data')
        if 'price' not in data:
            raise ValueError("Price data not found in the fetched data")
        data['price'] = str(round(float(data['price']),2))
        ti.xcom_push(key='transformed_data', value=data)
    except Exception as e:
        raise RuntimeError(f"Error transforming data: {e}")

def load(**kwargs):
    try:
        ti = kwargs['ti']
        data = ti.xcom_pull(key='transformed_data', task_ids='transform_btcusdt_data')
        conn = psycopg2.connect(
            database="Secret",
            user="Secret",
            password="Secret",
            host="Secret",
            port="Secret"
        )
        cursor = conn.cursor()
        cursor.execute("INSERT INTO solusdt_data (price) VALUES (%s)", (data['price'],))
        conn.commit()
        cursor.close()
        conn.close()
    except psycopg2.DatabaseError as e:
        raise RuntimeError(f"Error connecting to the database: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data into the database: {e}")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'binance_solusdt_etl_dag',
    default_args=default_args,
    description='A simple ETL DAG to fetch, transform, and store SOLUSDT data from Binance',
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
)

start_task = DummyOperator(task_id='start', dag=dag)

extract_task = PythonOperator(
    task_id='extract_btcusdt_data',
    python_callable=extract,
    provide_context=True,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_btcusdt_data',
    python_callable=transform,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_btcusdt_data',
    python_callable=load,
    provide_context=True,
    dag=dag,
)

end_task = DummyOperator(task_id='end', dag=dag)

start_task >> extract_task >> transform_task >> load_task >> end_task
