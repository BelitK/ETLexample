#!/bin/bash

read -r -p "Run Compose [R] or Create Tables [C] or Stop Compose [S] o  First Run [F]`echo $'\n> '`" choice 

if [[ ${choice} == [rR] ]]; then
	echo "building with compose"
	sudo docker compose up -d --build --remove-orphans || sudo docker compose up -d --build
elif [[ ${choice} == [sS] ]]; then
	echo "Stopping Compose"
	sudo docker compose down -v
elif [[ ${choice} == [fF] ]]; then
    echo "Running airflow init"
    sudo docker compose up airflow-init
elif [[ ${choice} == [cC] ]]; then

	echo "creating tables"
    source .env
    execute_sql_file() {
        cat $1 | sudo docker exec -i coin-postgres-1 psql -U $DB_USER -d $DB_NAME
    }
    read -p "enter folder name with sqls " SQL_FOLDER


    for sql in "$SQL_FOLDER"/*
    do
      echo "running $sql now"
      execute_sql_file $sql
    done
fi