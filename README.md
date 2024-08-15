# Crypto Coin Price Forecasting and Prediction

This project demonstrates how to create a simple ETL (Extract, Transform, Load) pipeline using Airflow, Python and PostgreSQL. The pipeline fetches BTCUSDT data from Binance, processes it, and stores it in a PostgreSQL database. Second part is forecasting and prediction on coin time series data using tensorflow with continuous learning and fastapi integration for interfacing with model.
Using streamlit for demo

# How To Use

```
bash run.sh
Run Compose [R] or Create Tables [C] or Stop Compose [S] o  First Run [F]
>
```

For first time usage airflow needs to be initialized with user info, after initialization Compose can be started and tables related to dags can be created.

## Airflow
Airflow UI can be accessed from localhost with 8090 port, related dag folder is '/dags', Except ticker data other use ETL format and all of them use XCom for data sharing between tasks, rate limiting implemented for api usage restrictions, all dags run every 5 minutes and related table sql script are in tables folder, every 10 minutes data digestion begins and data thats unprocessed by ai model accumulates in simplified_data table.

## Postgres

Postgres can be accessed from localhost with 5432 port, config can be found in postgres_config folder and can be personalized


## FastApi and Tensorflow
#TODO gonna add text
## Project Structure
#TODO gonna add text and uml
## TODOS

#TODO gonna add time series forecasting with tensorflow and fastapi
#TODO streamlit part of the project