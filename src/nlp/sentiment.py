from __future__ import annotations

from transformers import pipeline


def load_sentiment(model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
	return pipeline("sentiment-analysis", model=model_name)


def analyze_sentiment(analyzer, text: str):
	return analyzer(text)
