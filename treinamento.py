import cv2, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.python.framework import graph_util


path_train = "OID/Dataset/train"
path_train_p = "dataset/train"
epochs = 2

DATA_CLASSES = ['Bicycle', 'Bus', 'Car', 'Motorcycle', 'Person', 'Truck']

N_CLASSES = 6

for iclass in DATA_CLASSES:
    if iclass not in os.listdir(path_train_p):
        os.mkdir(path_train_p + "/" + iclass)


# Funções de pré-processamento
def resize(img):
    width = int(img.shape[1] * 50 / 100)
    height = int(img.shape[0] * 50 / 100)
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

# Parâmetros para o treinamento do modelo
def train_vgg_model():
    height = 224
    width = 224
    seed = 2
    batch_size = 20
    

    # Criação dos geradores de dados
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.1,
        preprocessing_function=tf.keras.applications.vgg19.preprocess_input
    )

    train_data = datagen.flow_from_directory(
        directory = path_train_p, # onde a pasta grande ta
        target_size = (height, width), # tamanho da imagem
        shuffle = True,
        seed = seed,
        batch_size = batch_size,
        subset = "training" # se as imagens vão ser pra treino ou validação
    )

    val_data = datagen.flow_from_directory(
        directory = path_train_p,
        target_size = (height, width),
        seed = seed,
        batch_size = batch_size,
        subset = "validation"
    )

    # Carregamento da arquitetura VGG pré-treinada
    base_model_vgg = VGG19(
        input_shape=(height, width, 3),
        include_top=False, 
        pooling = "average",
        weights="imagenet"
    )
    base_model_vgg.trainable = False
    
    return base_model_vgg, train_data, val_data


def generate_ssd_model(base_model, train_data, val_data):
    # Construção do modelo SSD com a arquitetura VGG
    x = Flatten()(base_model.output)
    x = Dense(256, activation="relu")(x)
    output = Dense(N_CLASSES, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Compilação do modelo
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"]
    )

    # Treinamento do modelo
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )
    
    # Salvando o modelo em formato .h5
    model.save("ssd_trained_model.h5")


def generate_cnn_model(base_model, train_data, val_data):
    model = tf.keras.models.Sequential(
        [base_model,
        tf.keras.layers.Flatten(), # transforma uma camada quadrada em um vetor
        tf.keras.layers.Dense(units = 256, activation = "relu"),
        tf.keras.layers.Dense(units = N_CLASSES, activation = "softmax") 
        ])

    model.compile(
        loss = CategoricalCrossentropy(), # quão bom ou ruim está o modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), 
        metrics = ["accuracy"] 
    )

    model.fit(train_data,
            validation_data = val_data,
            epochs = 2)

    model.save('cnn_trained_model.h5')

# Pré-processamento das imagens e salvando em novo diretório
prepare_dataset = input("Need prepare dataset? [Y/n]: ")
if prepare_dataset.lower() in ["y", "yes", ""]:
    pp_save(path_train, path_train_p)

base_model_vgg, train_data, val_data = train_vgg_model()
generate_cnn_model(base_model_vgg, train_data, val_data)
# generate_ssd_model(base_model_vgg, train_data, val_data)
