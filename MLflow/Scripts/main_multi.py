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
from multiprocessing import Pool, cpu_count

def optimize_model_wrapper(args):
    """Función envoltorio para optimize_models que acepta una tupla de argumentos."""
    model_name, X_train, y_train, X_val, y_val, n_trials = args
    return optimize_models(model_name, X_train, y_train, X_val, y_val, n_trials)

def main():
    # Parsear argumentos
    args = parse_arguments()
    n_trials = args.n_trials  # Ahora `n_trials` viene de la línea de comandos

    # Cargar datos
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    X_train, X_val, y_train, y_val = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Preprocesar los datos de entrenamiento y validación
    print("🔎 Preprocesando texto... 📖")
    with Pool() as pool:
        X_train = list(tqdm(pool.imap_unordered(preprocess_text, X_train), total=len(X_train), desc="Preprocesando train"))
        X_val = list(tqdm(pool.imap_unordered(preprocess_text, X_val), total=len(X_val), desc="Preprocesando val"))

    # Modelos a optimizar
    models_to_optimize = ["NaiveBayes", "LogisticRegression", "RandomForest"]

    # Optimizar modelos con multiprocessing y n_trials configurable
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(
            optimize_model_wrapper,  # Usar la función envoltorio
            [(model_name, X_train, y_train, X_val, y_val, n_trials) for model_name in models_to_optimize]  # Pasar argumentos como una tupla
        ), total=len(models_to_optimize), desc="Optimizando modelos"))

    # Combinar resultados de la optimización
    best_hyperparams = {}
    for model_name, hyperparams in zip(models_to_optimize, results):
        best_hyperparams[model_name] = hyperparams

    # Crear pipelines con los mejores hiperparámetros
    pipelines = create_pipelines(best_hyperparams)

    # Entrenar y evaluar modelos
    for model_name, pipeline in pipelines.items():
        accuracy, precision, recall, report, trained_pipeline = train_and_evaluate(model_name, pipeline, X_train, y_train, X_val, y_val)

        # Registrar en MLflow
        log_experiment(trained_pipeline, accuracy, precision, recall, best_hyperparams[model_name], model_name)

        print(f"\n📌 Modelo {model_name} registrado en MLflow con Accuracy: {accuracy}")

if __name__ == "__main__":
    main()