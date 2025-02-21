import nltk
import re 
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from bs4 import BeautifulSoup
import contractions
from spellchecker import SpellChecker


nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    
    Preprocesa el texto aplicando varias técnicas de limpieza y normalización.

    Args:
        text (str): Texto a preprocesar.

    Returns:
        str: Texto preprocesado.
    """
    
    

    # Manejar etiquetas HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Corregir errores ortográficos
    #spell = SpellChecker()
    #corrected_words = [spell.correction(word) or word for word in text.split()] # <-- Modificación aquí
    #text = " ".join(corrected_words)

    # Expandir contracciones
    text = contractions.fix(text)

    # Eliminar URLs
    text = re.sub(r'http\S+', '', text)

    # Eliminar URLs
    text = re.sub(r'http\S+', '', text)

    # Quitar signos de puntuación
    text = re.sub(r'[^\w\s]', '', text) 

    # Convertir a minúsculas
    text = text.lower()

    # Lematizar
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
    
    # Normalizar espacios en blanco
    text = re.sub(' +', ' ', text)  # Reemplazar múltiples espacios por uno solo

    # Convertir números a palabras
    text = re.sub(r'\b\d+\b', lambda m: num2words(int(m.group(0))), text)

    return text