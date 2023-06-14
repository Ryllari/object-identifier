import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

height = 224
width = 224
path_train = "OIDv4_ToolKit/OID/Dataset/train"

classes = os.listdir(path_train)

model = tf.keras.models.load_model('modelo.h5')


# coloque a sua foto aqui
# clica no icone de pasta <<<<- e arrasta uma imagem pra fazer o upload
image = tf.keras.preprocessing.image.load_img('images2.jpg', 
                                              target_size = (height,width))

image = tf.keras.preprocessing.image.img_to_array(image) # transforma uma imagem em um array
images = np.expand_dims(image, axis = 0) # transforma um array em um tensor
process_images = tf.keras.applications.vgg19.preprocess_input(np.copy(images))

predicts = model.predict(process_images)

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.imshow(images[0].astype(np.uint8))

plt.subplot(1,2,2)
plt.barh(classes, predicts[0])
plt.show()