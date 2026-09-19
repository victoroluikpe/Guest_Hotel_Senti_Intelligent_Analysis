import dagshub
import os
import mlflow
from dotenv import load_dotenv
load_dotenv(override = True)
from config.constant import DAGSHUB_REPO, DAGSHUB_USERNAME, experiment_name
# local accesibility or connection to mlflow
# def setup_mlflow_connection():
    

#     dagshub.init(
#         repo_owner=DAGSHUB_USERNAME,
#         repo_name=DAGSHUB_REPO,
#         mlflow=True
# )

# mlflow.set_experiment(experiment_name)
# production Environment MLFLOW ACCESS
def setup_mlflow_connection():
    dagshub_token = os.getenv("MLFLOW_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("Mlflow Token variable not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = DAGSHUB_USERNAME
    repo_name = DAGSHUB_REPO

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    # set experiment
    mlflow.set_experiment(experiment_name)