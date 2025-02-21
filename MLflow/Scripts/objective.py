from train_and_evaluate import train_and_evaluate
from create_pipelines import create_pipelines

def objective(trial, model_name, X_train, y_train, X_val, y_val):
    """
    Función objetivo para Optuna.

    Args:
        trial: Instancia de Optuna trial.
        model_name (str): Nombre del modelo.
        X_train, y_train: Datos de entrenamiento.
        X_val, y_val: Datos de validación.

    Returns:
        float: Precisión (accuracy) del modelo optimizado.
    """
    # Hiperparámetros de vectorización TF-IDF
    tfidf_params = {} 
    tfidf_params['max_df'] = trial.suggest_float('max_df', 0.5, 1.0)
    tfidf_params['min_df'] = trial.suggest_int('min_df', 1, 5)
    
    ngram_range_options = {
        1: (1, 1),
        2: (1, 2),
    }
    ngram_range_choice = trial.suggest_categorical("tfidf_ngram_range", [1, 2])
    tfidf_params['ngram_range'] = ngram_range_options[ngram_range_choice]

    # Espacios de búsqueda de hiperparámetros modelos
    hyperparams = {}
    if model_name == "NaiveBayes":
        hyperparams["alpha"] = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
    elif model_name == "LogisticRegression":
        hyperparams["C"] = trial.suggest_float("C", 1e-3, 10.0,log=True)
        hyperparams["max_iter"] = trial.suggest_int("max_iter", 100, 500)
    elif model_name == "RandomForest":
        hyperparams["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
        hyperparams["max_depth"] = trial.suggest_int("max_depth", 5, 30)

    # Crear pipeline y evaluar
    pipelines = create_pipelines({model_name: hyperparams}, tfidf_params)
    pipeline = pipelines[model_name]
    accuracy, _, _, _, _ = train_and_evaluate(model_name, pipeline, X_train, y_train, X_val, y_val)
    
    return accuracy