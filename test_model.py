import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.src.utils.sequence_utils import pad_sequences
from keras.src.utils.numerical_utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from helpers import get_sequences_and_labels, get_actions
from constants import MAX_LENGTH_FRAMES, DATA_PATH, MODELS_PATH, MODEL_NAME

# Definición de rutas
ROOT_PATH = os.getcwd()
DATA_TEST_H5_PATH = os.path.join(ROOT_PATH, "H5_test_final")
MODELS_PATH = os.path.join(ROOT_PATH, "models")

def test_model(data_path, model_path):
    actions = get_actions(data_path)  # ['word1', 'word2', 'word3']

    sequences, labels = get_sequences_and_labels(actions, data_path)
    
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES, padding='post', truncating='post', dtype='float32')
    
    X = np.array(sequences)
    y_true = np.array(labels)

    model = load_model(model_path)
    
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred_classes)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')
    report = classification_report(y_true, y_pred_classes, target_names=actions)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print('Classification Report:')
    print(report)

    return accuracy, precision, recall, f1, report

if __name__ == "__main__":
    data_path = DATA_TEST_H5_PATH  # Usar la carpeta H5_test para evaluar el modelo
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)  # Ruta del modelo entrenado
    
    test_model(data_path, model_path)