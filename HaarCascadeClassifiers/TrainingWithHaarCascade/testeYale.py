import cv2
import os
import numpy as np
from PIL import Image

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#reconhecedor = cv2.face.EigenFaceRecognizer_create()
#reconhecedor.read('classificadorEigenYale.yml')
#reconhecedor = cv2.face.FisherFaceRecognizer_create()
#reconhecedor.read('classificadorFisherYale.yml')
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read('classificadorLBPHYale.yml')


totalAcertos = 0
percentualAcerto = 0.0
totalConfianca = 0.0

caminhos = [os.path.join('yalefaces/yalefaces/teste', f) for f in os.listdir('yalefaces/yalefaces/teste')]

for caminhoImagem in caminhos:
    imagemFace = Image.open(caminhoImagem).convert('L')
    imagemNP = np.array(imagemFace, 'uint8')
    facesDetectadas = detectorFace.detectMultiScale(imagemNP)
    for (x, y, l, a) in facesDetectadas:
        idprev, confianca = reconhecedor.predict(imagemNP)
        idcorreto = int(os.path.split(caminhoImagem)[1].split('.')[0].replace('subject',''))
        print(f"{idcorreto} foi classificado como {idprev} com confiança {confianca}")
        if idcorreto == idprev:
            totalAcertos += 1
            totalConfianca += confianca

percentualAcerto = (totalAcertos / 30) * 100
totalConfianca = totalConfianca / totalAcertos
print(f"Percentual de acertos {percentualAcerto} com total de confiança {totalConfianca}")