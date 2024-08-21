# Crypto Coin Price Forecasting and Prediction

This project demonstrates how to create a simple ETL (Extract, Transform, Load) pipeline using Airflow, Python and PostgreSQL. The pipeline fetches BTCUSDT data from Binance, processes it, and stores it in a PostgreSQL database. Second part is forecasting and prediction on coin time series data using tensorflow (LSTM) with continuous learning and fastapi integration for interfacing with model.
Using streamlit for demo

# How To Use

```
bash run.sh
Run Compose [R] or Create Tables [C] or Stop Compose [S]
>
```

For first time usage airflow needs to be initialized with user info, after initialization Compose can be started and tables related to dags can be created.
!! Personally suggest wait until simplified_data table exceedes batch size and sequence length with 7/3 split because i didnt add check for data length 
## Bash Script
Simple script for initializing and running project.
- First run starts airflow-init and initializes airflow with postgres and migrates.
- [R] option starts project and checks FIRST_RUN param in .env file as true-false.
- [C] option creates tables with sql scripts in a folder that gets with input (tables/ as ex.)
- [S] stops project with -v param.
## Airflow
Airflow UI can be accessed from localhost with 8090 port, related dag folder is '/dags', Except ticker data other use ETL format and all of them use XCom for data sharing between tasks, rate limiting implemented for api usage restrictions, all dags run every 5 minutes and related table sql script are in tables folder, every 10 minutes data digestion begins and data thats unprocessed by ai model accumulates in simplified_data table.

## Postgres
Postgres can be accessed from localhost with 5432 port, config can be found in postgres_config folder and can be personalized
Airflow and Dags uses this as main database, crypto depth, price_data, ticker and simplified_data are connected to dags, simplified_data also is used as training data source for ai.

## FastApi and Tensorflow
Tensorflow model for forecast and prediction, Used LSTM as a alternative to transformer because of hardware, used 2 lstm layers with dropouts and dense for output layer. # todo can be better with auto hyper params and less complex 
Continuous learning is implemented with mse as score and trains with new data that get accumulated in simplified_data table, at start of learning a temp model is generated with new data and mse is compared with mse, if result passes new model is saved and used and table gets truncated.  
there is an endpoint for every part but im gonna automate unneeded parts.
## Spark
Runs as a part of stack and used in data operations on ai part.
# Streamlit
Developed to be a interface for interacting with api and model.
## Project Structure
- Postgres 
  1. ETL pipelines
  2. AI training
- Fastapi & Tensorflow
  1. AI training and stuff
  2. Data related stuff
  3. Model interface api
- Airflow
  1. ETL pipelines
  2. Data digestion
- Spark
  1. Data stuff
- Streamlit
  1. Model gui interface
  2. Data visualization
- Docker
  1. Containerization
  2. easier

## TODOS
- Gonna add realtime data to streamlit for representation and funs
- Update api and automate processes
- Add diagram image to structure part
- Gotta add data size check for batch

!! Disclaimer Im not responsible for any damage related to this project use or modify as you want