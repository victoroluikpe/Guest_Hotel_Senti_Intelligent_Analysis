from src.logger import configure_logger
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from src.pipeline.training_pipeline import Train_model
from src.pipeline.prediction_pipeline import predict_sentiment
from src.exception import MyException
from pydantic import BaseModel
import io
import sys



logging = configure_logger()

app = FastAPI(
    title = "Montra Guest Experience Sentiment Analysis API",
    description = "API for hotel guest review sentiment analysis, batch sentiment analysis and model training",
    version = "1.0.0",
)

@app.get("/")
def roo():
    return{
        "message": "montra experience and sentiment analysis API is up and running",
        "status": "active",
        "docs": "docs"
    }


class TextRequest(BaseModel):
    text: str

predictor = None
# load Model on startup
@app.on_event("startup")
async def load_model_on_startup():
    global predictor

    predictor = predict_sentiment()

    logging.info("sentiment model loaded successfully")

@app.post("/predict")
def predict_text(request: TextRequest):
    result = predictor.predict(request.text)

    top_label = max(result, key=lambda x:x["score"])
    # result = [{label: positive, score: 0.89}, {label: negative, score: 0.32}, {label: neutral, score: 0.10}]
    return {
        "label": top_label["label"],
        "confidence": float(top_label["score"])
    }


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """upload a csv with the review text column and get back prediction on all the csv file data """
    try:
        contents = await file.read()
        df = pd.read_csv(io.stringIO(contents.decode("utf-8")))

        if 'review_text' not in df.columns:
            return {"error": "csv file must have the review_text column"}

        results_list = []
        for idx, row in df.iterrows():
            try:
                review = str(row["review_text"])

                if not review.strip() or review.lower() == "nan":
                    raise ValueError("empty review text")

                result = predictor.predict(review)

                if result is None or len(result) == 0:
                    raise ValueError("empty result from the model")

                top_label = max(result, key=lambda x:x["score"])
                result_row = row.to_dict()
                result_row["sentiment_label"] = top_label["label"]
                result_row["sentiment_confidence"] = float(top_label["score"])

                result_row = {key: None if pd.isna(value) else value for key, value in result_row.items()}

                results_list.append(result_row)
            except Exception as e:
                result_row = row.to_dict()
                result_row["sentiment_label"] = f"Error: {str(e)[:50]}"
                result_row["sentiment_confidence"] = 0.0
                result_row = {key: None if pd.isna(value) else value for key, value in result_row.items()}
                results_list.append(result_row)

        return results_list
    except Exception as e:
        logging.error(f"batch prediction failed {e}")

@app.get("/train")
def train_model():
    try:
        Train_model()
        return {"status": "success", "message": "model training completed successfully"}
    except Exception as e:
        return {"status": "error", "message": f"training failed {e}"}
    
                









        

