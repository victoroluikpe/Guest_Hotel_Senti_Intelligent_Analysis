import mlflow
from mlflow.tracking import MlflowClient
import dagshub
from src.logger import configure_logger
import os
from src.utility.mlflow_setup import setup_mlflow_connection
from config.constant import experiment_name, classification_model_name
from src.exception import MyException

logging = configure_logger()
def get_existing_model(experiment_name = experiment_name):
    client = mlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        return None
    existing_model = sorted(
        runs,
        key = lambda x:x.data.metrics.get("eval_f1", 0),
        reverse = True
    )[0]

    return existing_model

def get_existing_model_f1_score(experiment_name = experiment_name):
    best_run = get_existing_model(experiment_name)
    if best_run is None:
        return None
    return best_run.data.metrics.get("eval_f1", 0)

def load_registered_model(model_name = classification_model_name):
    setup_mlflow_connection()
    model_uri = f"models:/{model_name}/latest"
    sentiment_pipeline = mlflow.transformers.load_model(model_uri)

    return sentiment_pipeline



    


