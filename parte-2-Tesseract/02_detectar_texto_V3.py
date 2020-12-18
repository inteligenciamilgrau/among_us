import cv2
import pytesseract
import numpy as np
try:
    from PIL import ImageGrab, Image
except ImportError:
    import Image

def mostraImagem(imagem):
    cv2.imshow("imagem", imagem)

# Coloque abaixo o local em que o tesseract está instalado no seu computador
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\SEU_USUARIO\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"

x = 40
y = 160
largura = 710
altura = 200

while True:
    larguraFinal = largura + x
    alturaFinal = altura + y

    imagem_Pil = ImageGrab.grab([x,y,larguraFinal,alturaFinal])
    imagem_Pil_np = np.array(imagem_Pil)

    img_gray = cv2.cvtColor(imagem_Pil_np, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(img_gray,200,255,cv2.THRESH_BINARY)

    mostraImagem(thresh1)

    # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
    # we need to convert from BGR to RGB format/mode:

    #configura = r'--oem 0 --psm 6 outputbase digits'
    configura = r'--oem 0 --psm 6'
    
    print(pytesseract.image_to_string(thresh1, config=configura))

    key = cv2.waitKey(30)
    print("key", key, "altura", altura, "largura", largura)
    if key == 13 or key == 27:  # 13 is the Enter Key 27 is ESC
        break
    elif key == 110:
        largura -= 10
        print(largura)
    elif key == 109:
        largura += 10
        print(largura)
cv2.destroyAllWindows()
