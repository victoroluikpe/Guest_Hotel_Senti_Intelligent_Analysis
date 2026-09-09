import dagshub
import os
import mlflow
from dotenv import load_dotenv
load_dotenv(override = True)
from config.constant import DAGSHUB_REPO, DAGSHUB_USERNAME, experiment_name
def setup_mlflow_connection():
    

    dagshub.init(
        repo_owner=DAGSHUB_USERNAME,
        repo_name=DAGSHUB_REPO,
        mlflow=True
)

mlflow.set_experiment(experiment_name)