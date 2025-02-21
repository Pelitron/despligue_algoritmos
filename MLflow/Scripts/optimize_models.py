import optuna
from objective import objective

def optimize_models(models, X_train, y_train, X_val, y_val, n_trials):
    """
    Optimiza múltiples modelos utilizando Optuna.

    Args:
        models (list): Lista de modelos a optimizar.
        X_train, y_train: Datos de entrenamiento.
        X_val, y_val: Datos de validación.
        n_trials (int): Número de iteraciones para Optuna.

    Returns:
        dict: Mejores hiperparámetros para cada modelo.
    """
    best_hyperparams = {}
    best_tfidf_params = {}
    
    for model_name in models:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model_name, X_train, y_train, X_val, y_val), n_trials=n_trials)
        best_hyperparams[model_name] = study.best_params
        best_tfidf_params[model_name] = {
            'max_df': study.best_params['max_df'],
            'min_df': study.best_params['min_df'],
            'ngram_range': (1, study.best_params['tfidf_ngram_range'])
            }

        best_hyperparams[model_name] = {k: v for k, v in study.best_params.items() if k not in ['max_df', 'min_df', 'tfidf_ngram_range']}
        

    print("✅ Best TF-IDF Params:", best_tfidf_params)
    print("✅ Best Model Params:", best_hyperparams)
    return best_hyperparams, best_tfidf_params