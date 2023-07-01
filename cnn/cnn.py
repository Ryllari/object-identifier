import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

height = 224
width = 224

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR, "cnn_trained_model.h5")
DATA_CLASSES = ['Car', 'Bus', 'Truck', 'Person', 'Motorcycle', 'Bicycle']

model = tf.keras.models.load_model(MODEL_PATH)


def identify_objects(img_path):
    image = tf.keras.preprocessing.image.load_img(
        img_path, 
        target_size=(height,width)
    )

    image = tf.keras.preprocessing.image.img_to_array(image) # transforma uma imagem em um array
    images = np.expand_dims(image, axis = 0) # transforma um array em um tensor
    process_images = tf.keras.applications.vgg19.preprocess_input(np.copy(images))

    predicts = model.predict(process_images)

    plt.figure(figsize = (12,5))
    plt.subplot(1,2,1)
    plt.imshow(images[0].astype(np.uint8))

    plt.subplot(1,2,2)
    plt.barh(DATA_CLASSES, predicts[0])
    plt.show()
