import os

def criar_pastas():
    path_train_p = "dataset"

    os.mkdir(path_train_p)

    sub = ['test','train','validation']
    sub2 = ['Car','Motorcycle','Bus', 'Truck', 'Bicycle', 'Person']

    for pasta in sub:
        os.mkdir(path_train_p + "/" + pasta)
        for pasta2 in sub2:
            os.mkdir(path_train_p + "/" + pasta+ "/"+ pasta2) 
