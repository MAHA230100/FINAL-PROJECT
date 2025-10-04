from __future__ import annotations

from transformers import MarianMTModel, MarianTokenizer


def load_translator(src_lang: str = "en", tgt_lang: str = "hi"):
	model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
	tokenizer = MarianTokenizer.from_pretrained(model_name)
	model = MarianMTModel.from_pretrained(model_name)
	return tokenizer, model


def translate(tokenizer, model, text: str) -> str:
	inputs = tokenizer([text], return_tensors="pt", padding=True)
	gen = model.generate(**inputs)
	return tokenizer.decode(gen[0], skip_special_tokens=True)
