#!/bin/bash

# Load environment variables
source .env

# Check if it's the first run
if [[ -z "${FIRST_RUN}" || "${FIRST_RUN}" == "true" ]]; then
    echo "First time run detected. Initializing Airflow."
    sudo docker compose up airflow-init

    # Set FIRST_RUN to false in the .env file
    sed -i 's/FIRST_RUN=true/FIRST_RUN=false/' .env
    echo "Initialization complete. FIRST_RUN set to false."

    echo "Building and starting Docker Compose"
    sudo docker compose up -d --build --remove-orphans || sudo docker compose up -d --build
else
    read -r -p "Run Compose [R] or Create Tables [C] or Stop Compose [S]`echo $'\n> '`" choice

    if [[ ${choice} == [rR] ]]; then
        echo "Building with compose"
        sudo docker compose up -d --build --remove-orphans || sudo docker compose up -d --build
    elif [[ ${choice} == [sS] ]]; then
        echo "Stopping Compose"
        sudo docker compose down -v
    elif [[ ${choice} == [cC] ]]; then
        echo "Creating tables"
        execute_sql_file() {
            cat $1 | sudo docker exec -i etlexample-postgres-1 psql -U $DB_USER -d $DB_NAME
        }
        read -p "Enter folder name with SQLs: " SQL_FOLDER

        for sql in "$SQL_FOLDER"/*
        do
          echo "Running $sql now"
          execute_sql_file $sql
        done
    fi
fi
