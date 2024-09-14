import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = "Data/3"
if not os.path.exists(folder):
    os.makedirs(folder)
counter = 0
take_picture = False
last_picture_time = 0
interval_time = 1  # Interval time in seconds
save_msg = "Saving picture: No"

while True:
    success, img = cap.read()

    if not success:
        print("Failed to read from camera.")
        break  # Exit the loop if failed to read from the camera

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the hand bounding box stays within the image dimensions
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(x + w + offset, img.shape[1]), min(y + h + offset, img.shape[0])
        imgCrop = img[y1:y2, x1:x2]

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("imagecrop", imgCrop)
        cv2.imshow("imagewhite", imgWhite)

        # Automatic picture taking
        current_time = time.time()
        if take_picture and current_time - last_picture_time >= interval_time:
            counter += 1
            filename = f'{folder}/Image_{time.strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
            cv2.imwrite(filename, imgWhite)
            print(f"Image saved: {filename}")
            last_picture_time = current_time

    cv2.putText(img, save_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("image", img)
    key = cv2.waitKey(1)

    # Start/Stop automatic picture taking
    if key == ord("s"):
        take_picture = True
        last_picture_time = time.time()
        save_msg = "Saving picture: Yes"
    elif key == ord("p"):
        take_picture = False
        save_msg = "Saving picture: No"

    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()

