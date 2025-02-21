from fastapi import APIRouter
from hf_pipelines import sentiment_analysis, text_generation, translate_text, summarize_text

router = APIRouter()

@router.get("/ping")
def ping():
    return {"message": "pong"}

@router.get("/sentiment")
def analyze_sentiment(text: str):
    result = sentiment_analysis(text)
    return {"sentiment": result}

@router.get("/generate")
def generate_text(prompt: str):
    result = text_generation(prompt)
    return {"generated_text": result}

@router.get("/translate")
def translate(text: str, target_lang: str = "es"):
    result = translate_text(text, target_lang)
    return {"translation": result}

@router.get("/summarize")
def summarize(text: str):
    result = summarize_text(text)
    return {"summary": result}