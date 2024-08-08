import aiohttp
import asyncio
import time
import logging
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_url(session, url):
    max_retries = 5
    backoff_factor = 2
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        retry_after = int(retry_after)
                    else:
                        retry_after = backoff_factor**attempt
                    logger.warning(
                        f"Rate limit exceeded. Retrying after {retry_after} seconds..."
                    )
                    await asyncio.sleep(retry_after)
                elif response.status >= 400:
                    response.raise_for_status()
                else:
                    return await response.json()
        except aiohttp.ClientResponseError as e:
            logger.error(f"Request failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff_factor**attempt)
            else:
                raise
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff_factor**attempt)
            else:
                raise
    raise Exception(f"Failed to fetch {url} after {max_retries} attempts")


async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        return await fetch_url(session, url)


def fetch_data_sync(url):
    return asyncio.run(fetch_data(url))

def generate_conn(conn_type = 'psql'):
    try:
        if conn_type == 'psql':
            conn = psycopg2.connect(
                database="airflow",
                user="airflow",
                password="airflow",
                host="postgres",
                port="5432",
            )
        elif conn_type == 'alch':
            # Define the PostgreSQL URL
            postgresql_url = 'postgresql://airflow:airflow@postgres:5432/airflow'

            # Create an engine
            conn = create_engine(postgresql_url).connect().connection
        else :
            raise RuntimeError('incorrect type selection')
        return conn
    except psycopg2.DatabaseError as e:
        raise RuntimeError(f"Error connecting to the database: {e}")
    except Exception as e:
        raise RuntimeError(f"Error  {e}")

def get_data(table):
    try:
        conn = generate_conn(conn_type='alch')
        return pd.read_sql(f'Select * from {table} where processed = False;',conn)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

def insert_data(table, data, conn_type='psql'):
    try:
        if conn_type == 'psql':
            conn = generate_conn()
            cursor = conn.cursor()
            placeholders = ', '.join([a for a in data.keys()])
            values = [a for a in data.values()]
            esses = ', '.join(['%s'] * len(data))
            print(placeholders)
            cursor.execute(f"INSERT INTO {table} ({placeholders}) VALUES ({esses})", values)
            conn.commit()
            cursor.close()
            conn.close()
        elif conn_type=='alch':
            conn = generate_conn()
            cursor = conn.cursor()

            columns = ", ".join(data.columns)
            values = ", ".join(["%s"] * len(data.columns))
            insert_statement = f"INSERT INTO {table} ({columns}) VALUES ({values})"
            # Insert DataFrame data into the PostgreSQL table
            for _, row in data.iterrows():
                cursor.execute(insert_statement, tuple(row))
            conn.commit()
            cursor.close()
            conn.close()
    except psycopg2.DatabaseError as e:
        raise RuntimeError(f"Error connecting to the database: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading data into the database: {e}")


def update_data(table, data):
    try:
        conn = generate_conn()
        cursor = conn.cursor()
        for id in data:
            cursor.execute(f"update {table} set processed = True where id={id}")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Error updating data in the database: {e}")