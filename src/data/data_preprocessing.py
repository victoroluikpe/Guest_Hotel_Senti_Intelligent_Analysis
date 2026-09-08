from src.logger import configure_logger
from src.exception import MyException
import sys

import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from src.data.data_cleaning import Data_cleaner

logging = configure_logger()


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels = labels

    def __len__(self):
      return len(self.labels)

    def __getitem__(self, idx):
      item= {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
      item['labels'] = torch.tensor(self.labels[idx])
      return item


class SentimentAnalysisProcessor:

    def __init__(self, model_name = "distilbert-base-multilingual-cased", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def split_data(self, dataframe):
        """split data into training and testing"""
        try:
            X = dataframe["cleaned_review_text"].astype(str) # full text
            y = dataframe["sentiment_label"]

            # Build a stable label mapping ONCE use it everywhere
            label2id ={"negative":0, "neutral": 1, "positive": 2}
            id2label = {v: k for k, v in label2id.items()}
            y_encoded = y.map(label2id)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=y_encoded #keep class balance consistent across splits
            )

            X_train = X_train.tolist() # this are transformerModel they uselist
            X_test = X_test.tolist()
            y_train = y_train.tolist()
            y_test = y_test.tolist()

            return X_train, X_test, y_train, y_test, label2id, id2label

        except Exception as e:
           logging.error(f"error occured while splitting the data into training and testing {e}")
           raise MyException(e, sys)

    def tokenize_data(self, X_train, X_test):
        """Tokenizing the training and testing dataset for transformer model to use"""
        try:
            train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length = self.max_length)
            test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length = self.max_length)
            logging.info(f"sucessfully tokenized our training and testing dataset")
            return train_encodings, test_encodings

        except Exception as e:
            logging.error(f"error occured while tokenizing the dataset {e}")
            raise MyException(e, sys)

    def create_dataset(self, train_encodings, test_encodings, y_train, y_test):
        try:
            train_dataset = SentimentDataset(train_encodings, y_train)
            test_dataset = SentimentDataset(test_encodings, y_test)
            logging.info(f"successfully created the sentiment dataset....")

            return train_dataset, test_dataset
        except Exception as e:
             logging.error(f"error occured while created the dataset {e}")
             raise MyException(e, sys)


def DataProcessor():
    cleaned_customer_review_data = Data_cleaner()
    SentimentAnalysisEngineProcessor = SentimentAnalysisProcessor()
    X_train, X_test, y_train, y_test, label2id, id2label = SentimentAnalysisEngineProcessor.split_data(cleaned_customer_review_data)
    train_encodings, test_encodings = SentimentAnalysisEngineProcessor.tokenize_data(X_train, X_test)
    train_dataset, test_dataset = SentimentAnalysisEngineProcessor.create_dataset(train_encodings, test_encodings, y_train, y_test)
    return train_dataset, test_dataset, label2id, id2label

#DataProcessor()
      

        