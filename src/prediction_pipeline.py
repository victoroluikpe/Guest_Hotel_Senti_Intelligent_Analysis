from src.utility.model_loader import load_registered_model

class predict_sentiment:
    def __init__(self,):
        self.pipeline = load_registered_model()

    def predict(self, text):
        raw_result = self.pipeline(text)
        return raw_result
