import tensorflow as tf
from keras.losses import CategoricalCrossentropy
import os
from pdi import *

pdi_train = False 
pdi_test = False
pdi_validation = False

classes = os.listdir(path_train)
n_classes=len(classes)

if pdi_train:
    pp_save(path_train, path_train_p)
if pdi_test:
    pp_save(path_test, path_test_p)
if pdi_validation:
    pp_save(path_validation, path_validation_p)

height = 224
width = 224
seed = 2

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split = 0.1,
    # vc passa essa linha debaixo pro seu input ficar do mesmo jeito que 
    # o da pessoa que treinou a rede convolucional
    preprocessing_function = tf.keras.applications.vgg19.preprocess_input
)

train_data = datagen.flow_from_directory(
    directory = path_train_p, # onde a pasta grande ta
    target_size = (height, width), # tamanho da imagem
    shuffle = True,
    seed = seed,
    batch_size = 20,
    subset = "training" # se as imagens vão ser pra treino ou validação
)

val_data = datagen.flow_from_directory(
    directory = path_train_p,
    target_size = (height, width),
    seed = seed,
    batch_size = 20,
    subset = "validation"
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split = 0.1)

test_data = test_datagen.flow_from_directory(
    directory = path_train_p,
    target_size = (height, width),
    shuffle = True,
    seed = seed,
    batch_size = 20,
    subset = "validation"
)

# camadas convolucionais 
# responsaveis por extrair caracteristicas da imagem que são importantes pra classificação
# modelo pré-treinado VGG19 (Transfer learning )
base_model = tf.keras.applications.VGG19(input_shape = (height, width, 3),
                                         include_top = False,
                                         pooling = "average",
                                         weights = "imagenet")
base_model.trainable = False

model = tf.keras.models.Sequential(
    [base_model,
     tf.keras.layers.Flatten(), # transforma uma camada quadrada em um vetor
     tf.keras.layers.Dense(units = 256, activation = "relu"),
     tf.keras.layers.Dense(units = n_classes, activation = "softmax") 
     ])

model.compile(
    loss = CategoricalCrossentropy(), # quão bom ou ruim está o modelo
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), 
    metrics = ["accuracy"] 
)

model.fit(train_data,
          validation_data = val_data,
          epochs = 2)

model.save('modelo.h5')