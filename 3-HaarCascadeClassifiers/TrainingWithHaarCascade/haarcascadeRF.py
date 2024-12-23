import cv2
import numpy as np

camera = cv2.VideoCapture(0)

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier('haarcascade_eye.xml')
amostra = 1
numeroAmostras = 25

id = input('Digite seu identificador: ' )

largura, altura = 220, 220

print("Capturando as faces...")

while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #print(np.average(imagemCinza))
    facesDetectadas = classificador.detectMultiScale(imagem, scaleFactor=1.5,minSize=(150,150))
   
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem,
                       (x, y),
                       (x+l, y+a), 
                        (0, 0, 255),
                        2
        )
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinza = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhos = classificadorOlho.detectMultiScale(regiaoCinza)

        for (olhoX, olhoY, olhoL, olhoA) in olhos:
            cv2.rectangle(regiao, (olhoX, olhoY), (olhoX + olhoL, olhoY + olhoA), (0, 255, 0), 2)
            


            if cv2.waitKey(1) & 0xFF==ord('q'):
                if np.average(imagemCinza) > 110: #Trabalhando luminosidade através da média do nivel de cinza
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura) )
                    cv2.imwrite("fotos/pessoa."+str(id)+"."+str(amostra)+".jpg", imagemFace)
                    print("Foto tirada com sucesso"+str(amostra))
                    amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >=numeroAmostras +1 ):
        break



print("Faces capturadas")
camera.release()
cv2.destroyAllWindows()