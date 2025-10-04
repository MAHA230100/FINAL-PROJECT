from __future__ import annotations

from typing import List

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def load_clinical_ner(model_name: str = "dslim/bert-base-NER"):
	# For clinical tasks consider: "emilyalsentzer/Bio_ClinicalBERT" with fine-tuning
	tok = AutoTokenizer.from_pretrained(model_name)
	mdl = AutoModelForTokenClassification.from_pretrained(model_name)
	ner = pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")
	return ner


def run_ner(ner_pipeline, text: str) -> List[dict]:
	return ner_pipeline(text)
