import cv2
import os
import numpy as np



def getImagemComID():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(12)
    return np.array(ids), faces

ids, faces = getImagemComID()

print('treinando...')

#num_components = numero de componentes principais (Eigenfaces baseados em PCA) 50 é um bom numero conforme a documentação
#Threshold é o parametro de limiar. Quanto menor, menor deve ser a distancia da imagem atual com a da base de dados
eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold = 1.7976931348623157e+308)
fisherface = cv2.face.FisherFaceRecognizer_create(num_components=50)
lbph = cv2.face.LBPHFaceRecognizer_create()

eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')


fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento realizado')