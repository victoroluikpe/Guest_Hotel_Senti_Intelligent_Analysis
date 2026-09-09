from src.logger import configure_logger
from src.exception import MyException
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv(override=True) # importin database
import pandas as pd
import os
import sys

logging = configure_logger() # defin function


class DataIngestion:
    def __init__(self):
        self.engine = create_engine(
            os.getenv("DATABASE_URL"),
            pool_pre_ping = True,
            pool_recycle = 1800
        )

    def test_connection(self):
        """test the postgresql connection"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT version();"))
                version = result.fetchone()

                logging.info(f"successfully connected to the database.")
                logging.info(f"postgreSQL version: {version[0]}")

                return True
        except Exception as e:
            logging.error(f"Error connecting to the database: {e}")
            raise MyException(e, sys)

    def load_table(self, table_name: str) -> pd.DataFrame:
        """load a complte table into a pandas dataframe"""
        try:
            query = text(f"SELECT * FROM {table_name}")

            dataframe = pd.read_sql(query, self.engine)

            logging.info(f"loaded {table_name}, it contains {dataframe.shape[0]} rows and {dataframe.shape[1]} columns")

            return dataframe
        except Exception as e:
            logging.error(f"error occured while connecting to the database")
            raise MyException(e, sys)

    def load_all_data(self):
        try:
            """load all the required data from the database"""
            hotels_detail = self.load_table("hotels")
            guest_data = self.load_table("guests")
            customer_review = self.load_table("reviews")
            bookings_data = self.load_table("bookings")

            logging.info(hotels_detail.head())
            logging.info(f"data successfully loaded")

            return {
                "hotels": hotels_detail,
                "guests": guest_data,
                "reviews": customer_review,
                "bookings": bookings_data
            }
        except Exception as e:
            logging.error(f"error occured while loading all dataset {e}")
            raise MyException(e, sys)

    def data_merger(self):
        """this function is used for merging all the dataframe together as one complete dataset"""
        try:
            data = self.load_all_data()
            customer_review = data["reviews"]
            hotels_detail = data["hotels"]
            bookings_data = data["bookings"]
            guests_data = data["guests"]

            customer_review_data = customer_review.merge(bookings_data, on=["booking_id", "guest_id", "hotel_id"], how="inner")
            customer_review_data = customer_review_data.merge(guests_data, on="guest_id", how="inner")
            customer_review_data = customer_review_data.merge(hotels_detail, on="hotel_id", how="inner")

            logging.info(f"the dataset has been successfully merged{customer_review_data.shape}")
            return customer_review_data
        except Exception as e:
            logging.error("error occured while merging the data {e}")
            raise MyException(e, sys)

    def closs(self):
        self.engine.dispose()

def data_loader():
    # ingestion = DataIngestion()
    # if not ingestion.test_connection():
    #     raise SystemExit("unable to connect to the database ")
    #customer_review_data = ingestion.data_merger()
    customer_review_data = pd.read_csv(r"C:/Users/pc/Downloads/Guest_Hotel_Senti_Intelligent_Analysis/Testing_data_sample.csv")
    return customer_review_data
        

#data_loader()

    



