from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from ..ai_tools.summarizer import create_summarizer
from ..ai_tools.classifier import create_classifier

router = APIRouter(prefix="/ai-tools", tags=["ai-tools"])


class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 150


class ClassifyRequest(BaseModel):
    text: str


@router.post("/summarize")
def summarize_text(req: SummarizeRequest):
    """Summarize input text using AI"""
    summarizer = create_summarizer()
    summary = summarizer.summarize(req.text, req.max_length)
    keywords = summarizer.extract_keywords(req.text)
    
    return {
        "summary": summary,
        "keywords": keywords,
        "original_length": len(req.text),
        "summary_length": len(summary)
    }


@router.post("/classify")
def classify_text(req: ClassifyRequest):
    """Classify input text using AI"""
    classifier = create_classifier()
    label, confidence = classifier.classify(req.text)
    
    return {
        "label": label,
        "confidence": confidence,
        "text": req.text
    }


@router.get("/available")
def get_available_tools():
    """Get list of available AI tools"""
    return {
        "tools": [
            {"name": "summarizer", "description": "Text summarization"},
            {"name": "classifier", "description": "Text classification"},
            {"name": "translator", "description": "Text translation (coming soon)"},
            {"name": "sentiment", "description": "Sentiment analysis (coming soon)"}
        ]
    }
