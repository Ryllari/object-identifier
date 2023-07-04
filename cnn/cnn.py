import cv2
import numpy as np
import os
import tensorflow as tf

from PIL import Image
import requests
from stqdm import stqdm

height = 224
width = 224

DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR, "cnn_trained_model.h5")
DATA_CLASSES = ['Car', 'Bus', 'Truck', 'Person', 'Motorcycle', 'Bicycle']

def download_yolo_weights(url="http://vpn.service.app.br:442/files/cnn_trained_model.h5"):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # Tamanho do bloco para atualização da barra de progresso
    with open(MODEL_PATH, 'wb') as file:
        with stqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=MODEL_PATH) as t:
            for data in response.iter_content(block_size):
                file.write(data)
                t.update(len(data))
    print("Download concluído. O arquivo foi salvo em:", MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    download_yolo_weights()

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
