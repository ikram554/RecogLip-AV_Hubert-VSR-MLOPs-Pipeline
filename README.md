# RecogLip-AV_Hubert-VSR-MLOPs-Pipeline
Welcome to the RecogLip Git repository for the AV_Hubert VSR MLOPs Pipeline! This repository houses the codebase and resources for our cutting-edge lip recognition Raspberry Pi Headgear.


Run these commands to create a docker image



## Installation

```
  docker build -t av_hubert .

  docker run -it --rm av_hubert
```
    
### Note:
If you're considering using the Jenkins pipeline, please keep in mind the following important instructions for customization:
- To add your DockerHub credentials, navigate to Manage Jenkins → Manage Credentials in Jenkins, and update the appropriate field.
- To replace the GitHub URL in the checkout stage, simply update it to reflect your own repository's URL.
- Finally, ensure that you update the "registry" field to match the URL for your own DockerHub repository.
By following these steps, you can easily customize the Jenkins pipeline for your own purposes.

## Apache Airflow Installation
- Install Docker and Docker Compose
- Linux
```
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.6.1/docker-compose.yaml'
```
- Windows
```
Invoke-WebRequest -Uri "https://airflow.apache.org/docs/apache-airflow/2.0.2/docker-compose.yaml" -OutFile docker-compose.yaml
```
- Initilize docker compose
```
docker compose up airflow-init 
```
- Run
```
docker compose up -d
```


## Run Jenkinsfile
- Install Jenkins
- Once Jenkins is installed, access it through your web browser by visiting http://localhost:8080 
- In the pipeline job configuration, go to the "Pipeline" section.
- Select "Pipeline script from SCM" as the definition.
- Paste the content of this repo Jenkinsfile in it
- Save the job configuration.
- Click on "Build now"

## DVC Setup


```
pip install dvc
```

```
dvc init
```

```
dvc add models/finetune-model.pt
```

```
dvc commit
```

```
dvc push
```

```
git push
```
