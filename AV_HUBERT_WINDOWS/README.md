
# Project Title

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

## Run Jenkinsfile
- Install Jenkins
- Once Jenkins is installed, access it through your web browser by visiting http://localhost:8080 
- In the pipeline job configuration, go to the "Pipeline" section.
- Select "Pipeline script from SCM" as the definition.
- Paste the content of this repo Jenkinsfile in it
- Save the job configuration.
- Click on "Build now"
