from transformers import pipeline
import torch


class SentimentModel:
    def __init__(self, device='cpu'):
        self._sentiment_analysis = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

    def predict(self, text):
        return self._sentiment_analysis(text)[0]["label"]

#if __name__ == "__main__":
#    sample_text = "The Dow Jones Industrial Average (^DJI) turned red."
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#    model = SentimentModel(device=device)
#    sentiment = model.predict(text=sample_text)
#    print(sentiment)
