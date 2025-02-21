from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
generation_pipeline = pipeline("text-generation", model="gpt2")
translation_pipeline = pipeline("translation_en_to_fr")
summarization_pipeline = pipeline("summarization")

def sentiment_analysis(text: str):
    return sentiment_pipeline(text)[0]

def text_generation(prompt: str):
    return generation_pipeline(prompt, max_length=50)[0]["generated_text"]

def translate_text(text: str, target_lang: str):
    return translation_pipeline(text)[0]['translation_text']

def summarize_text(text: str):
    return summarization_pipeline(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']