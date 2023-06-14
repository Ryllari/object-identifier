import cv2, os
import matplotlib.pyplot as plt

path_train = "../OID/Dataset/train"
path_test = "../OID/Dataset/test"
path_validation = "../OID/Dataset/validation"

path_train_p = "../dataset/train"
path_test_p = "../dataset/test"
path_validation_p = "../dataset/validation"

classes = os.listdir(path_train)
n_classes=len(classes)

os.mkdir(path_train_p)
os.mkdir(path_test_p)
os.mkdir(path_validation_p)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

def resize(img):
    
    width = int(img.shape[1] * 50 / 100)
    height = int(img.shape[0] * 50 / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def preprocessamento(img):
    img = resize(img)
    img = grayscale(img)
    img = equalize(img) # Converter em escala de cinza e Padronizar a Luminosidade das imagens
    return img

def pp_save(path_origem, path_destino):  
    count = 0
    # Obtendo imagens de treino e aplicando pre-processamento
    for _ in range (0,n_classes):
        arquivos = os.listdir(path_origem+"/"+classes[count])
        x = 0
        for arq in arquivos:
            print(arq)
            if x > 5000:
                break
            else:
                curImg = cv2.imread(path_origem+"/"+classes[count]+"/"+arq)
                curImg = preprocessamento(curImg) 
                cv2.imwrite(os.path.join(path_destino+"/"+classes[count], arq), curImg)
            x +=1

        count +=1

def plot_dataset(dataset):

    plt.gcf().clear()
    plt.figure(figsize = (15, 15))

    for features, labels in dataset.take(1):

        for i in range(3):

            plt.subplot(3, 3, i + 1)
            plt.axis('off')

            plt.imshow(features[i].numpy().astype('uint8'))
            plt.title(classes[labels[i]])
            plt.show()

