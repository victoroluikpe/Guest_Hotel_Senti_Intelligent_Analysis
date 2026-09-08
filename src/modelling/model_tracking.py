import mlflow
import sys
import mlflow.transformers
from src.logger import configure_logger
from src.exception import MyException
from transformers import pipeline
from src.utility.model_loader import get_existing_model_f1_score
from config.constant import model_name, classification_model_name, training_args
from src.utility.mlflow_setup import setup_mlflow_connection

from dotenv import load_dotenv
load_dotenv(override= True)
logging = configure_logger()

class ModelTracker:
    def __init__(self, experiment_name = experiment_name):
        try:
            setup_mlflow_connection()
            self.experimentt = experiment_name
        except Exception as e:
            logging.error(f"error occured while initalizing mlflow {e}")
            raise MyException(e, sys)
    def model_update_tracker(self, trainer, metrics):
        try:
            new_f1 = metrics["eval_f1"] 
            old_f1 = get_existing_model_f1_score(self.experiment_name)

            print(f"previous model f1 csore: {old_f1}")
            print(f"new model f1: {new_f1}")

            if old_f1 is None or new_f1 > old_f1:
                with mlflow.start_run(run_name="sentiment-classification") as run:

                    print("Run ID:", run.info.run_id)
                    # Log evaluation metrics
                    mlflow.log_metric("eval_accuracy",metrics["eval_accuracy"])
                    mlflow.log_metric("eval_f1",metrics["eval_f1"])
                    mlflow.log_metric("eval_loss",metrics["eval_loss"])

                    # Log training parameters
                    mlflow.log_param("num_epochs",training_args.num_train_epochs)
                    mlflow.log_param("batch_size",training_args.per_device_train_batch_size)
                    mlflow.log_param("learning_rate",training_args.learning_rate)

                    sentiment_pipeline = pipeline(
                        task="text-classification", # Corrected typo here
                        model=trainer.model,
                        tokenizer=model_name, # Changed from trainer.tokenizer to tokenizer
                        return_all_scores=True
                    )
                    # Log and register model
                    mlflow.transformers.log_model(
                        transformers_model=sentiment_pipeline,
                        name="sentiment_model",
                        registered_model_name="classification_model_namel"
                    )
                logging.info(f"new model and tokenizer have been registered in MLFLOW ")

            else:
                logging.info(f"Model not pushed (performance didn"t improve)")

        except Exception as e:
            logging.error(f"error occured while pushing the model to mlflow {e}")
            raise MyException(e, sys)
        




    