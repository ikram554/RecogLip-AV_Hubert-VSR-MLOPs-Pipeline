# NOTE: These are test dags for submission as many small unit functions of data cleaning, data pre-processing and model training steps donot apply to our FYP project

from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def task1():
    print("Task 1")

def task2():
    print("Task 2")

def task3():
    print("Task 3")

def task4():
    print("Task 4")

def task5():
    print("Task 5")

def task6():
    print("Task 6")

default_args = {
    'owner': 'Ikram Khan',
    'start_date': datetime(2023, 6, 1),
}

dag = DAG(
    'multi_downstream_dag',
    default_args=default_args,
    schedule_interval=None,
)

with dag:
    t1 = PythonOperator(
        task_id='task1',
        python_callable=task1,
    )

    t2 = PythonOperator(
        task_id='task2',
        python_callable=task2,
    )

    t3 = PythonOperator(
        task_id='task3',
        python_callable=task3,
    )

    t4 = PythonOperator(
        task_id='task4',
        python_callable=task4,
    )

    t5 = PythonOperator(
        task_id='task5',
        python_callable=task5,
    )

    t6 = PythonOperator(
        task_id='task6',
        python_callable=task6,
    )

    t1 >> t2
    t1 >> t3
    t2 >> t4
    t3 >> t4
    t4 >> t5
    t4 >> t6
