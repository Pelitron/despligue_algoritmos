from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np

def train_and_evaluate(model_name, pipeline, X_train, y_train, X_val, y_val):
    """
    Entrena y evalúa un modelo.

    Args:
        model_name (str): Nombre del modelo.
        pipeline: Pipeline con el modelo optimizado.
        X_train, y_train: Datos de entrenamiento.
        X_val, y_val: Datos de validación.

    Returns:
        tuple: (accuracy, precision, recall, report, modelo entrenado).
    """
    print(f"\n🚀 Entrenando {model_name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    present_classes = np.unique(y_pred)  # Obtener las clases presentes en las predicciones
    present_classes_mask = np.isin(y_val, present_classes)  # Máscara para las clases presentes en y_val
    y_val_filtered = y_val[present_classes_mask]  # Filtrar y_val
    y_pred_filtered = y_pred[present_classes_mask]  # Filtrar y_pred
    
    accuracy = accuracy_score(y_val_filtered, y_pred_filtered)
    precision = precision_score(y_val_filtered, y_pred_filtered, average='weighted')
    recall = recall_score(y_val_filtered, y_pred_filtered, average='weighted')
    report = classification_report(y_val_filtered, y_pred_filtered)
    
    print(f"\n✅ Modelo {model_name} evaluado con Accuracy: {accuracy}")
    return accuracy, precision, recall, report, pipeline