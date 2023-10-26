import cv2
import pytesseract
from PIL import Image

img_cv = cv2.imread(r'text.png')
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img_rgb))
cv2.imshow("Image", img_cv)
cv2.waitKey(10000)

