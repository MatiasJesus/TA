import os
import pandas as pd
from constants import ROOT_PATH

# Definición de rutas
H5_TRAIN_PATH = os.path.join(ROOT_PATH, "H5_train")
H5_TEST_PATH = os.path.join(ROOT_PATH, "H5_test")
H5_TRAIN_AUX_PATH = os.path.join(ROOT_PATH, "H5_train_aux")
H5_TEST_AUX_PATH = os.path.join(ROOT_PATH, "H5_test_aux")
H5_TRAIN_FINAL_PATH = os.path.join(ROOT_PATH, "H5_train_final")
H5_TEST_FINAL_PATH = os.path.join(ROOT_PATH, "H5_test_final")

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def combine_h5_files(h5_file1, h5_file2, output_file):
    """
    Combina dos archivos H5 en un solo archivo.

    :param h5_file1: Ruta al primer archivo H5.
    :param h5_file2: Ruta al segundo archivo H5.
    :param output_file: Ruta al archivo H5 combinado de salida.
    """
    # Leer los archivos H5
    data1 = pd.read_hdf(h5_file1, key='data') if os.path.exists(h5_file1) else pd.DataFrame([])
    data2 = pd.read_hdf(h5_file2, key='data') if os.path.exists(h5_file2) else pd.DataFrame([])
    
    # Combinar los DataFrames
    combined_data = pd.concat([data1, data2], ignore_index=True)
    
    # Guardar el DataFrame combinado en un nuevo archivo H5
    combined_data.to_hdf(output_file, key='data', mode='w')

def process_and_combine_folders(h5_train_path, h5_train_aux_path, h5_train_final_path, h5_test_path, h5_test_aux_path, h5_test_final_path):
    create_folder(h5_train_final_path)
    create_folder(h5_test_final_path)
    
    # Obtener archivos comunes en H5_train y H5_train_aux
    train_files = set(os.listdir(h5_train_path)).intersection(set(os.listdir(h5_train_aux_path)))
    test_files = set(os.listdir(h5_test_path)).intersection(set(os.listdir(h5_test_aux_path)))
    
    for train_file in train_files:
        if train_file.endswith('.h5'):
            word = train_file.replace('.h5', '')
            train_hdf_path = os.path.join(h5_train_path, f"{word}.h5")
            train_aux_hdf_path = os.path.join(h5_train_aux_path, f"{word}.h5")
            train_final_hdf_path = os.path.join(h5_train_final_path, f"{word}.h5")
            
            print(f'Combinando archivos H5 de entrenamiento para la palabra "{word}"...')
            combine_h5_files(train_hdf_path, train_aux_hdf_path, train_final_hdf_path)
            print(f"Archivos combinados para la palabra {word}!")

    for test_file in test_files:
        if test_file.endswith('.h5'):
            word = test_file.replace('.h5', '')
            test_hdf_path = os.path.join(h5_test_path, f"{word}.h5")
            test_aux_hdf_path = os.path.join(h5_test_aux_path, f"{word}.h5")
            test_final_hdf_path = os.path.join(h5_test_final_path, f"{word}.h5")
            
            print(f'Combinando archivos H5 de prueba para la palabra "{word}"...')
            combine_h5_files(test_hdf_path, test_aux_hdf_path, test_final_hdf_path)
            print(f"Archivos combinados para la palabra {word}!")

if __name__ == "__main__":
    process_and_combine_folders(H5_TRAIN_PATH, H5_TRAIN_AUX_PATH, H5_TRAIN_FINAL_PATH,
                                H5_TEST_PATH, H5_TEST_AUX_PATH, H5_TEST_FINAL_PATH)
