import cv2
import os
import numpy as np
from PIL import Image



def getImagemComID():
    caminhos = [os.path.join('yalefaces/yalefaces/treinamento', f) for f in os.listdir('yalefaces/yalefaces/treinamento')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split('.')[0].replace('subject',''))
        ids.append(id)
        faces.append(imagemNP)
      
    return np.array(ids), faces

ids, faces = getImagemComID()

print('treinando...')

#num_components = numero de componentes principais (Eigenfaces baseados em PCA) 50 é um bom numero conforme a documentação
#Threshold é o parametro de limiar. Quanto menor, menor deve ser a distancia da imagem atual com a da base de dados
eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000)
fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000)
lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 50)

eigenface.train(faces, ids)
eigenface.write('classificadorEigenYale.yml')


fisherface.train(faces, ids)
fisherface.write('classificadorFisherYale.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPHYale.yml')

print('Treinamento realizado')