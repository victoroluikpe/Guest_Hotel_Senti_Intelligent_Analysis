import sys
from src.data.data_ingection import data_loader
import re
from src.logger import configure_logger
from src.exception import MyException


logging = configure_logger()

class DataCleaning:
    def clean_review(self, text) -> str:
        # convert to string
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<,*?>', ' ', text)
        # Remove URLs
        text = text.encode('ascii', 'ignore').decode('ascii')
        # convert to lowercase
        text = text.lower()
        # remove special characters
        # keep letters, numbers and saces
        text = re.sub(r'[^a-z0-9\s]', ' ',text)
        # remove extra whitespace
        text = re.sub(r'\s+',' ',text).strip()

        return text

    def clean_review_column(self, dataframe):
        """apply that cleaning using our clean review function on the customer review text """
        try:
            customer_review_data = dataframe.copy()
            customer_review_data["cleaned_review_text"] = customer_review_data["review_text"].map(self.clean_review)
            logging.info(customer_review_data.head())

            return customer_review_data
        except Exception as e:
            logging.error(f"error occured during data cleaning {e}")
            raise MyException(e, sys)

def Data_cleaner():
    try:
        customer_data = data_loader()
        DataCleaningEngine = DataCleaning()
        cleaned_customer_review_data = DataCleaningEngine.clean_review_column(customer_data)

        return cleaned_customer_review_data
    except Exception as e:
        logging.error(f"error occured during data cleaning {e}")
        raise MyException(e, sys)

#Data_cleaner()