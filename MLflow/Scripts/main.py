import mlflow
from arguments import parse_arguments
from optimize_models import optimize_models
from create_pipelines import create_pipelines
from train_and_evaluate import train_and_evaluate
from text_preprocessing import preprocess_text
from log_experiment import log_experiment
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool

def main():
    # Parsear argumentos
    args = parse_arguments()
    n_trials = args.n_trials
    experiment_name = args.experiment_name
    print(f"📌 Experimento: {experiment_name} | N-Trials: {n_trials}")

    # Cargar datos
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    X_train, X_val, y_train, y_val = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Preprocesar los datos de entrenamiento y validación
    print("🔎 Preprocesando texto... 📖")
    X_train = [preprocess_text(text) for text in tqdm(X_train, desc="Preprocesando train")]
    X_val = [preprocess_text(text) for text in tqdm(X_val, desc="Preprocesando val")]

    # Modelos a optimizar
    models_to_optimize = ["NaiveBayes", "LogisticRegression", "RandomForest"]

    # Optimizar modelos con `n_trials` configurable
    best_hyperparams, best_tfidf_params = optimize_models(models_to_optimize, X_train, y_train, X_val, y_val, n_trials)

    # Crear pipelines con los mejores hiperparámetros
    pipelines = {}
    for model_name in models_to_optimize:
        pipeline = create_pipelines({model_name: best_hyperparams[model_name]}, best_tfidf_params[model_name])[model_name] 
        pipelines[model_name] = pipeline 

    # Entrenar y evaluar modelos
    for model_name, pipeline in pipelines.items():
        accuracy, precision, recall, report, trained_pipeline = train_and_evaluate(model_name, pipeline, X_train, y_train, X_val, y_val)

    # Registrar en MLflow
        log_experiment(trained_pipeline, accuracy, precision, recall, best_hyperparams[model_name],best_tfidf_params[model_name], model_name, experiment_name, n_trials)

        print(f"\n📌 Modelo {model_name} registrado en MLflow con Accuracy: {accuracy}")

if __name__ == "__main__":
    main()