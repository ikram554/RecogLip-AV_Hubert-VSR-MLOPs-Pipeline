pipeline {
    agent any 
    environment {
        registry = "ikramkhan1/av_hubert_mlops:v1"
        registryCredential = 'dockerhub_id'
        dockerImage = ''
    }
    
    stages {
        stage('Cloning Git') {
            steps {
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[url: 'https://github.com/ikram554/RecogLip-AV_Hubert-VSR-MLOPs-Pipeline.git']])
            }
        }
    
        stage('Building image') {
          steps{
            script {
              dockerImage = docker.build registry
            }
          }
        }
    
        stage('Upload Image') {
         steps{    
             script {
                docker.withRegistry( '', registryCredential ) {
                dockerImage.push()
                }
            }
          }
        }
    }
}
