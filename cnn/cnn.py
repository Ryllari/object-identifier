import cv2
import numpy as np
import os
import tensorflow as tf

from PIL import Image
from utils import download_file_from_google_drive

height = 224
width = 224

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR, "cnn_trained_model.h5")
DATA_CLASSES = ['Car', 'Bus', 'Truck', 'Person', 'Motorcycle', 'Bicycle']

model = tf.keras.models.load_model(MODEL_PATH)

def local_identify_objects(img_path_file, output_dir):
    image, _ = identify_objects(img_path_file, from_web=False)
    output_filename = os.path.join(output_dir, "cnn-result.jpg")
    cv2.imwrite(output_filename, image)
    print(f"Resultado salvo em: {output_filename}")
    return image

def identify_objects(img_path, from_web=True):
    if from_web:
        img = cv2.imread(img_path.name)
    else:
        img = cv2.imread(img_path)

    image = tf.keras.preprocessing.image.load_img(
        img_path, 
        target_size=(height,width)
    )

    image = tf.keras.preprocessing.image.img_to_array(image) # transforma uma imagem em um array
    images = np.expand_dims(image, axis = 0) # transforma um array em um tensor
    process_images = tf.keras.applications.vgg19.preprocess_input(np.copy(images))

    predicts = model.predict(process_images)

    y_ssd = {i: [] for i in DATA_CLASSES}
    confs_array = predicts[0]
    class_indexes = np.where(predicts > 0.5)[1]
    for i in class_indexes:
        class_name = DATA_CLASSES[i]
        conf = confs_array[i]
        y_ssd[class_name].append(conf)

    return img, y_ssd
