#############################
# YOLO object detection 1.0 #
# Developed by Leandro Rego #
#  contato@leandrorego.com  #
#############################

# inportar bibliotecas
import os
import cv2
import requests
import numpy as np

from PIL import Image
from stqdm import stqdm


DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(DIR, "yolov3.cfg")
WEIGHTS = os.path.join(DIR, "yolov3.weights")
CLASSES = os.path.join(DIR, "classes.names")
OUTPUT_FILE = "yolo/yolov3.weights"
DATA_CLASSES = ['Car', 'Bus', 'Truck', 'Person', 'Motorcycle', 'Bicycle']


def download_yolo_weights(url="https://pjreddie.com/media/files/yolov3.weights"):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # Tamanho do bloco para atualização da barra de progresso
    with open(WEIGHTS, 'wb') as file:
        with stqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=OUTPUT_FILE) as t:
            for data in response.iter_content(block_size):
                file.write(data)
                t.update(len(data))
    print("Download concluído. O arquivo foi salvo em:", WEIGHTS)


if not os.path.exists(WEIGHTS):
    download_yolo_weights()

# Carregar a rede YOLO
net = cv2.dnn.readNet(WEIGHTS, CONFIG)

# Carregar nomes das classes
classes = []
with open(CLASSES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Gerar cores aleatórias para cada classe
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def local_identify_objects(img_path_file, output_dir):

    # Carregar imagem de entrada
    input_image = cv2.imread(img_path_file)
    
    image, _ = identify_objects(input_image, from_web=False)
    output_filename = os.path.join(output_dir, "yolo-result.jpg")
    cv2.imwrite(output_filename, image)
    print(f"Resultado salvo em: {output_filename}")
    return image

def identify_objects(img, from_web=True):
    if from_web:
        img = Image.open(img)
        img = np.array(img)

    # Obter as dimensões da imagem
    height, width, channel = img.shape

    # Normalizar imagem e criar um blob de entrada
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Definir as camadas de saída desejadas
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Inicializar listas para caixas delimitadoras, confianças e IDs de classes
    boxes = []
    confidences = []
    class_ids = []

    # Iterar sobre cada camada de saída
    for output in layer_outputs:
        # Iterar sobre cada detecção
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            class_name = classes[class_id].capitalize()
            confidence = scores[class_id]
            if confidence > 0.5 and class_name in DATA_CLASSES:
                # Escalar as coordenadas da caixa delimitadora para o tamanho da imagem
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Calcular as coordenadas do retângulo da caixa delimitadora
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Adicionar as coordenadas da caixa delimitadora, confiança e ID da classe às listas
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar supressão não máxima para eliminar caixas sobrepostas
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    image = generate_image(img, boxes, indices, confidences, class_ids)
    return image

def generate_image(image, boxes, indices, confidences, class_ids):
    # Verificar se há pelo menos uma detecção
    y_yolo = {i: [] for i in DATA_CLASSES}
    if len(indices) > 0:

        for i in indices.flatten():
            # Obter as coordenadas da caixa delimitadora
            x, y, w, h = boxes[i]
            # Obter a classe e confiança da detecção
            class_id = class_ids[i]
            label = classes[class_id].capitalize()
            confidence = confidences[i]
            # Desenhar a caixa delimitadora e exibir o rótulo da classe
            color = colors[class_id]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            y_yolo[label].append(confidence)
            
    # Saída
    return image,y_yolo