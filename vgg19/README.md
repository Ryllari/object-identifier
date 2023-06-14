# Rede CNN para classificação de Imagens

Rede Neural Convolucional criada para reconhecimento de imagens utilizada como trabalho pratico para disciplina de Processamento Digital de Imagens do mestrado em ciência da computação - PPgCC UFERSA/UERN


# Download do dataset

Para realizar o download do dataset, se fez necessário a utilização do [OIDv4 ToolKit](https://github.com/EscVM/OIDv4_ToolKit), que consiste em um kit de ferramentas para obtenção de datasets a partir do  [Open Image Dataset](https://storage.googleapis.com/openimages/web/index.html).

### Como usar:

- clonar repositório do OIDv4 ToolKit
```
git clone https://github.com/EscVM/OIDv4_ToolKit.git
```
- Entrar na pasta clonada
```
cd OIDv4_ToolKit
```
- Executar o comado para download do dataset
```
python main.py downloader --classes classes.txt --type_csv train --limit 5000 --image_IsInside 0 --image_IsOccluded 0
```
- Observações

	- classes.txt: arquivo de texto onde é definido as classes das imagens que desejamos baixar, no nosso caso: carro, moto, caminhão, ônibus, bicicleta e pedestres
	- type_csv: tipo de imagens e sua função (treino, validação e teste)
	- limit: Limite de imagens que serão baixadas por classe
	- image_IsInside: Indica uma foto tirada de dentro do objeto (por exemplo, interior de um carro ou dentro de um edifício).
	- image_IsOccluded: Indica que o objeto está ocluído por outro objeto na imagem.
	
Após a execução dos passos anteriores, será criada uma estrutura parecida com a estrutura a seguir:
```

main_folder
│   cnn.py
│
└──OIDv4_ToolKit
		│   main.py
		│
		└───OID
		    │   file011.txt
		    │   file012.txt
		    │
		    └───csv_folder
		    |    │   class-descriptions-boxable.csv
		    |    │   validation-annotations-bbox.csv
		    |
		    └───Dataset
		        |
		        └─── test
		        |
		        └─── train
		        |
		        └─── validation
		             |
		             └───Car
		             |     |
		             |     |0fdea8a716155a8e.jpg
		             |     |2fe4f21e409f0a56.jpg
		             |     |...
		             |     └───Labels
		             |            |
		             |            |0fdea8a716155a8e.txt
		             |            |2fe4f21e409f0a56.txt
		             |            |...
		             |
		             └───Bus
		                   |
		                   |0b6f22bf3b586889.jpg
		                   |0baea327f06f8afb.jpg
		                   |...
		                   └───Truck
		                          |
		                          |0b6f22bf3b586889.txt
		                          |0baea327f06f8afb.txt
		                          |...

```

# Treinamento

# Importação do modelo