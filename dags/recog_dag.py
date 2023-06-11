# NOTE: These are test dags for submission as many small unit functions of data cleaning, data pre-processing and model training steps donot apply to our FYP project

from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def download_data():
    print("Downloading data...")

def clean_data():
    print("Cleaning data...")

def analyze_data():
    print("Analyzing data...")

def store_results():
    print("Storing results...")

default_args = {
    'owner': 'Ikram',
    'start_date': datetime(2023, 6, 1),
}

dag = DAG(
    'data_processing_workflow',
    default_args=default_args,
    schedule_interval='@daily',
)

with dag:
    task1 = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
    )

    task2 = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )

    task3 = PythonOperator(
        task_id='analyze_data',
        python_callable=analyze_data,
    )

    task4 = PythonOperator(
        task_id='store_results',
        python_callable=store_results,
    )

    task1 >> task2 >> task3 >> task4
