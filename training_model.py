import os
import numpy as np
from model import NUM_EPOCH, get_model
from keras.src.utils.sequence_utils import pad_sequences
from keras.src.utils.numerical_utils import to_categorical
from helpers import get_actions, get_sequences_and_labels
from constants import MAX_LENGTH_FRAMES, MODEL_NAME

# Definición de rutas
ROOT_PATH = os.getcwd()
DATA_TRAIN_H5_PATH = os.path.join(ROOT_PATH, "H5_train_final")
MODELS_PATH = os.path.join(ROOT_PATH, "models")

def training_model(data_path, model_path):
    actions = get_actions(data_path)  # ['word1', 'word2', 'word3']
    
    sequences, labels = get_sequences_and_labels(actions, data_path)
    
    sequences = pad_sequences(sequences, maxlen=MAX_LENGTH_FRAMES, padding='post', truncating='post', dtype='float32')

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    model = get_model(len(actions))
    model.fit(X, y, epochs=NUM_EPOCH)
    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    # Asegurarse de que la carpeta de modelos exista
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    
    data_path = DATA_TRAIN_H5_PATH  # Usar la carpeta H5_train para entrenar el modelo
    model_path = os.path.join(MODELS_PATH, MODEL_NAME)
    
    training_model(data_path, model_path)
    