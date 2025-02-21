import argparse

def parse_arguments():
    """
    Define y parsea los argumentos de entrada.

    Returns:
        argparse.Namespace: Argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos con MLflow y Optuna")
    parser.add_argument("--experiment_name", type=str, default="Clasificación_MLflow", help="Nombre del experimento en MLflow")
    parser.add_argument("--n_trials", type=int, default=30, help="Número de iteraciones para Optuna")
    
    return parser.parse_args()