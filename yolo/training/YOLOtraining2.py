import os
import shutil

# Configuração de pasta e arquivo
train_folder = '../../dataset/Dataset/train/'  # Caminho para a pasta de imagens e anotações de treinamento
data_file = 'trained.weights'  # Caminho para o arquivo de configuração dos dados
config_file = './yolo.cfg'  # Caminho para o arquivo de configuração do modelo

# Obter nomes de classes
class_names = [folder_name for folder_name in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, folder_name))]

# Criação do arquivo de configuração dos dados
data_config = f"""
classes = {len(class_names)}
train = {train_folder}/train.txt
valid = {train_folder}/valid.txt
names = {train_folder}/obj.names
backup = backup/
"""

with open(data_file, 'w') as f:
    f.write(data_config)

# Criação do arquivo de lista de imagens de treinamento
train_list_file = f'{train_folder}/train.txt'
valid_list_file = f'{train_folder}/valid.txt'

with open(train_list_file, 'w') as f:
    for class_name in class_names:
        class_folder = os.path.join(train_folder, class_name)
        for root, _, files in os.walk(class_folder):
            for filename in files:
                if filename.endswith('.jpg'):
                    image_path = os.path.join(root, filename)
                    label_folder = os.path.join(train_folder, class_name, 'Label')
                    label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt'))
                    f.write(f'{image_path}\n')
                    shutil.copy2(label_path, label_folder)  # Copiar anotações para pasta de treinamento

# Criação do arquivo de nomes das classes
names_file = f'{train_folder}/obj.names'
with open(names_file, 'w') as f:
    for class_name in class_names:
        f.write(f'{class_name}\n')

# Execute o treinamento do modelo YOLO usando o Darknet
# Comando para iniciar o treinamento
command = f'darknet detector train {data_file} {config_file}'

# Execute o comando para iniciar o treinamento
os.system(command)
