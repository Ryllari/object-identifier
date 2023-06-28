from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score

#Utiliza y_true como a lista de verdadeiros e y_yolo,y_cnn,y_ssd como as previsões do modelo
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

def to_label_indices(y_yolo, y_cnn, y_ssd):
    y_yolo = [list(classNames.keys())[list(classNames.values()).index(label)] for label in y_yolo]
    y_cnn = [list(classNames.keys())[list(classNames.values()).index(label)] for label in y_cnn]
    y_ssd = [list(classNames.keys())[list(classNames.values()).index(label)] for label in y_ssd]
    return y_yolo, y_cnn, y_ssd

def comparation_models(y_true, y_yolo, y_cnn, y_ssd):
    to_label_indices(y_yolo, y_cnn, y_ssd)

    # Cálculo da matriz de confusão
    cm_yolo = confusion_matrix(y_true, y_yolo)
    cm_cnn = confusion_matrix(y_true, y_cnn)
    cm_sdd = confusion_matrix(y_true, y_ssd)
    
    # Cálculo da sensibilidade
    recall_yolo = recall_score(y_true, y_yolo, average=None, labels=classNames.values())
    recall_cnn = recall_score(y_true, y_cnn, average=None, labels=classNames.values())
    recall_sdd = recall_score(y_true, y_ssd, average=None, labels=classNames.values())
    
    # Cálculo da precisão
    precision_yolo = precision_score(y_true, y_yolo, average=None, labels=classNames.values())
    precision_cnn = precision_score(y_true, y_cnn, average=None, labels=classNames.values())
    precision_sdd = precision_score(y_true, y_ssd, average=None, labels=classNames.values())
    
    # Cálculo da acurácia
    accuracy_yolo = accuracy_score(y_true, y_yolo)
    accuracy_cnn = accuracy_score(y_true, y_cnn)
    accuracy_sdd = accuracy_score(y_true, y_ssd)
    
    # Imprimir resultados
    print("Matriz de Confusão - YOLO:\n", cm_yolo)
    print("Recall - YOLO: {:.2f}".format(recall_yolo))
    print("Precisão - YOLO: {:.2f}".format(precision_yolo))
    print("Acurácia - YOLO: {:.2f}".format(accuracy_yolo))
    
    print("\nMatriz de Confusão - CNN:\n", cm_cnn)
    print("Recall - CNN: {:.2f}".format(recall_cnn))
    print("Precisão - CNN: {:.2f}".format(precision_cnn))
    print("Acurácia - CNN: {:.2f}".format(accuracy_cnn))
    
    print("\nMatriz de Confusão - SDD:\n", cm_sdd)
    print("Recall - SDD: {:.2f}".format(recall_sdd))
    print("Precisão - SDD: {:.2f}".format(precision_sdd))
    print("Acurácia - SDD: {:.2f}".format(accuracy_sdd))


    