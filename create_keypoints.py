import os
import pandas as pd
from sklearn.model_selection import train_test_split
from mediapipe.python.solutions.holistic import Holistic
from helpers import get_keypoints, insert_keypoints_sequence
from constants import ROOT_PATH

# Definición de rutas
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions_aux")
H5_TRAIN_PATH = os.path.join(ROOT_PATH, "H5_train_aux")
H5_TEST_PATH = os.path.join(ROOT_PATH, "H5_test_aux")

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_keypoints(frames_path, save_path):
    '''
    Crear keypoints para una palabra.
    '''
    data = pd.DataFrame([])
    
    with Holistic() as model_holistic:
        for n_sample, sample_name in enumerate(os.listdir(frames_path), 1):
            sample_path = os.path.join(frames_path, sample_name)
            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(data, n_sample, keypoints_sequence)

    data.to_hdf(save_path, key="data", mode="w")

def process_folder(frame_actions_path, h5_train_path, h5_test_path, test_size=0.3):
    create_folder(h5_train_path)
    create_folder(h5_test_path)
    words_path = os.path.join(ROOT_PATH, frame_actions_path)
    
    for word_name in os.listdir(words_path):
        word_path = os.path.join(words_path, word_name)
        train_hdf_path = os.path.join(h5_train_path, f"{word_name}.h5")
        test_hdf_path = os.path.join(h5_test_path, f"{word_name}.h5")
        
        # Verificar si los archivos H5 ya existen
        if os.path.exists(train_hdf_path) and os.path.exists(test_hdf_path):
            print(f'Archivos H5 para "{word_name}" ya existen, saltando...')
            continue

        temp_hdf_path = os.path.join(ROOT_PATH, f"{word_name}.h5")
        
        # Crear keypoints y guardarlos en un archivo temporal H5
        print(f'Creando keypoints de "{word_name}"...')
        create_keypoints(word_path, temp_hdf_path)
        print(f"Keypoints creados!")

        # Leer el archivo temporal y dividirlo en train/test
        data = pd.read_hdf(temp_hdf_path, key='data')
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

        # Guardar los datos divididos en sus respectivas carpetas
        train_data.to_hdf(train_hdf_path, key='data', mode='w')
        test_data.to_hdf(test_hdf_path, key='data', mode='w')

        # Eliminar el archivo temporal
        os.remove(temp_hdf_path)

if __name__ == "__main__":
    process_folder(FRAME_ACTIONS_PATH, H5_TRAIN_PATH, H5_TEST_PATH)
            
  #  # GENERAR SOLO DE UNA PALABRA
  #  word_name = "venir"
  #  word_path = os.path.join(words_path, word_name)
  #  hdf_path = os.path.join(DATA_PATH, f"{word_name}.h5")
  #  create_keypoints(word_path, hdf_path)
  #  print(f"Keypoints creados!")
