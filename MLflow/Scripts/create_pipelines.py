from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def create_pipelines(hyperparams, tfidf_params):
    """
    Crea pipelines con hiperparámetros optimizados.

    Args:
        hyperparams (dict): Diccionario con hiperparámetros para cada modelo.
        tfidf_params (dict): Hiperparámetros para TfidfVectorizer.

    Returns:
        dict: Pipelines optimizados para cada modelo.
    """
    pipelines = {}

    for model_name, params in hyperparams.items():
        if model_name == "NaiveBayes":
            model = MultinomialNB(**params)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(**params)
        elif model_name == "RandomForest":
            model = RandomForestClassifier(**params)

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words='english', **tfidf_params)),
            ("classifier", model)
        ])
        
        pipelines[model_name] = pipeline

    return pipelines