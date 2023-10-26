import time

import pytesseract
import cv2
import numpy as np
haarcascade = "model/haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("car.mp4")

cap.set(3, 1920)
cap.set(4, 1080)


min_area = 500

while True:
    success, img = cap.read()
    if success == True:
        # img = cv2.imread('car.jpg')
        plate_cascade = cv2.CascadeClassifier(haarcascade)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h

            if area > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Number", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                # time.sleep(0.1)
                img_roi = img[y: y+h, x: x+w]
                # cv2.imshow("NUMBER", img_roi)

                # img_1 = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                # img_2 = cv2.threshold(img_1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                # img_3 = cv2.medianBlur(img_1, 5)

                scale_percent = 220  # percent of original size
                width = int(img_roi.shape[1] * scale_percent / 100)
                height = int(img_roi.shape[0] * scale_percent / 100)
                dim = (width, height)
                img_roi = cv2.resize(img_roi, dim, interpolation=cv2.INTER_AREA)

                # get grayscale image
                def get_grayscale(image):
                    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


                # noise removal
                def remove_noise(image):
                    return cv2.medianBlur(image, 5)


                # thresholding
                def thresholding(image):
                    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


                # dilation
                def dilate(image):
                    kernel = np.ones((5, 5), np.uint8)
                    return cv2.dilate(image, kernel, iterations=1)


                # erosion
                def erode(image):
                    kernel = np.ones((5, 5), np.uint8)
                    return cv2.erode(image, kernel, iterations=1)


                # opening - erosion followed by dilation
                def opening(image):
                    kernel = np.ones((5, 5), np.uint8)
                    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


                # canny edge detection
                def canny(image):
                    return cv2.Canny(image, 100, 200)


                gray = get_grayscale(img_roi)
                thresh = thresholding(gray)
                opening = opening(gray)
                canny = canny(gray)

                cv2.imshow("NUM", opening)
                # time.sleep(0.1)
                num_car = pytesseract.image_to_string(opening)
                print(num_car)




        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('e'):
            break
    else:
        break