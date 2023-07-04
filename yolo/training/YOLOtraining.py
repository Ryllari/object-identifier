import cv2
import os
import numpy as np
from keras import backend as K
from keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.python.framework import graph_util
import tensorflow as tf
from yolov3_tf2.models import YoloV3, YoloLoss, yolo_anchors 

# Caminho para o diretório de treinamento das imagens
path_train = "../../dataset/Dataset/train"
# Caminho para o diretório de treinamento pré-processado
path_train_p = "../../dataset/trained"
# Número de etapas de treinamento
epochs = np.int32(int(2))

# Classes das imagens de treinamento
DATA_CLASSES = ['Bicycle', 'Bus', 'Car', 'Motorcycle', 'Person', 'Truck']
# Número total de classes
N_CLASSES = len(DATA_CLASSES)

# Criando diretórios para cada classe
for iclass in DATA_CLASSES:
    if iclass not in os.listdir(path_train_p):
        os.mkdir(path_train_p + "/" + iclass)

# Funções de pré-processamento
def resize(img):
    max_dim = max(img.shape[0], img.shape[1])
    scale = 416 / max_dim
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def preprocessamento(img):
    img = resize(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertendo para o formato RGB
    img = img / 255.0  # Normalização dos pixels entre 0 e 1
    return img

def pp_save(path_origem, path_destino):  
    for iclass in DATA_CLASSES:
        if iclass not in os.listdir(path_destino):
            os.mkdir(os.path.join(path_destino, iclass))
        
        class_path_origem = os.path.join(path_origem, iclass)
        class_path_destino = os.path.join(path_destino, iclass)
        
        count = 0
        for filename in os.listdir(class_path_origem):
            if count > 199:
                break
            else:
                img_path = os.path.join(class_path_origem, filename)
                curImg = cv2.imread(img_path)
                if str(curImg) != "None":
                    curImg = preprocessamento(curImg)
                    cv2.imwrite(os.path.join(class_path_destino, filename), curImg)
                    count += 1

# Verificando se é necessário pré-processar o dataset
prepare_dataset = input("Need to prepare the dataset? [Y/n]: ")
if prepare_dataset.lower() in ["y", "yes", ""]:
    pp_save(path_train, path_train_p)

# Criação dos geradores de dados
def train_data_generator(batch_size):
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1,
        preprocessing_function=tf.keras.applications.vgg19.preprocess_input
    )

    train_data = data_generator.flow_from_directory(
        directory=path_train_p,
        target_size=(416, 416),
        batch_size=batch_size,
        subset='training'
    )

    val_data = data_generator.flow_from_directory(
        directory=path_train_p,
        target_size=(416, 416),
        batch_size=batch_size,
        subset='validation'
    )

    return train_data, val_data

# Treinamento do modelo YOLO
def train_yolo_model():
    batch_size = 8
    train_data, val_data = train_data_generator(batch_size)

    yolo = YoloV3(classes=N_CLASSES)
    
    # Compilação do modelo
    yolo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=YoloLoss(anchors=yolo_anchors)
    )

    # Treinamento do modelo
    yolo.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    # Salvando o modelo em formato .h5
    yolo.save("yolo_trained_model.h5")

# Treinando e gerando o modelo YOLO
train_yolo_model()

