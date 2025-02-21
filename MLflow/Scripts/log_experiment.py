import mlflow
from mlflow.models.signature import infer_signature

def log_experiment(model, accuracy, precision, recall, hyperparams, 
    tfidf_params, model_name, experiment_name, n_trials, input_example=None):
    """
    Registra los resultados del modelo en MLflow.

    Args:
       model: Modelo entrenado.
        accuracy (float): Precisión del modelo.
        precision (float): Precisión media ponderada.
        recall (float): Recall medio ponderado.
        hyperparams (dict): Hiperparámetros optimizados del modelo.
        tfidf_params (dict): Parámetros optimizados de TfidfVectorizer.
        model_name (str): Nombre del modelo.
        experiment_name (str): Nombre del experimento.
        trial_number (int): Número del trial dentro del experimento.
        input_example: Ejemplo de entrada para la firma del modelo.
    """
    mlflow.set_experiment(experiment_name)

    signature = infer_signature(input_example, model.predict([input_example])[2]) if input_example is not None else None

    run_name = f"{model_name}_trial_{n_trials}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("trial_number", n_trials)
        mlflow.set_tag("experiment_name", experiment_name)
        mlflow.log_params(hyperparams)
        for key, value in tfidf_params.items():
            mlflow.log_param(f"tfidf_{key}", value)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example)
