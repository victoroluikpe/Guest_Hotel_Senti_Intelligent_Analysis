
from src.modelling.model_training import ModelTraining
from src.data.data_preprocessing import DataProcessor
from src.logger import configure_logger
from src.exception import MyException
from src.modelling.model_tracking import ModelTracker
import sys


logging = configure_logger()
def Train_model():
    try:
        train_dataset, test_dataset, label2id, id2label = DataProcessor()
        training_engine = ModelTraining(train_dataset, test_dataset, label2id, id2label)
        trainer = training_engine.train()
        training_engine.model_evaluation(trainer)
        results = training_engine.model_evaluation(trainer)
        logging.info(f"{results}")

        # pushing model to mlflow
        model_tracker = ModelTracker()
        model_tracker.model_update_tracker(trainer, results)
        return trainer, results
    except Exception as e:
        logging.error(f"pipeline error:{e}")
        raise MyException(e, sys)

#Train_model()

# b4 you run go to model_training comment out all the last def model_trainer()