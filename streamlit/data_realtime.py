import psycopg2
import pandas as pd

def generate_conn():
    conn = psycopg2.connect(
        dbname="airflow",
        user="airflow",
        password="airflow",
        host="postgres",
        port="5432"
    )
    return conn

def fetch_data_from_db(conn, table_name):
    query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 100"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
