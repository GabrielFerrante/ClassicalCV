import cv2


detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

fisherface = cv2.face.FisherFaceRecognizer_create()
fisherface.read('classificadorFisher.yml')

largura = 220
altura = 220

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL

camera = cv2.VideoCapture(0)

pessoas = ['gabriel', 'cae', 'jonas', 'kawes', 'vivian', 'flor','luiza', 'hugo']
print(pessoas.index('gabriel'))

while(True):

    conectado, imagem = camera.read()

    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5,minSize=(110,110))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem,
                       (x, y),
                       (x+l, y+a), 
                        (0, 0, 255),
                        2
        )
        id, confianca = fisherface.predict(imagemFace)

        cv2.putText(imagem, pessoas[id-1], (x,y+(a+30)), fonte, 2, (0,0,255))
        cv2.putText(imagem, str(confianca), (x,y+(a+50)), fonte, 2, (0,0,255))

    cv2.imshow('Face', imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
