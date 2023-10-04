# Multiple color detection in real time (green & magenta)

import cv2
import numpy as np

# Specifying upper and lower ranges of green color to detect in HSV format
lower_green = np.array([25, 40, 40])
upper_green = np.array([90, 255, 255])

# Specifying upper and lower ranges of magenta color to detect in HSV format
lower_magenta = np.array([136, 87, 111])
upper_magenta = np.array([180, 255, 255])

# Capturing webcam footage
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    success, video = cam.read()  # Reading webcam footage

    img_hsv = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)  # Convert BGR image to HSV format

    # Masking the image to find the green color
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # Masking the image to find the magenta color
    magenta_mask = cv2.inRange(img_hsv, lower_magenta, upper_magenta)

    # Finding contours in the green mask image
    green_contours, _ = cv2.findContours(
        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Finding contours in the magenta mask image
    magenta_contours, _ = cv2.findContours(
        magenta_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Finding position of all green contours
    if len(green_contours) != 0:
        for green_contour in green_contours:
            if cv2.contourArea(green_contour) > 500:
                x, y, w, h = cv2.boundingRect(green_contour)
                (x,y), radius = cv2.minEnclosingCircle(green_contour)
                center = (int(x),int(y))
                radius = int(radius)
                cv2.circle(video,center,radius,(0,255,0),3) # Drawing green circles

    # Finding position of all magenta contours
    if len(magenta_contours) != 0:
        for magenta_contour in magenta_contours:
            if cv2.contourArea(magenta_contour) > 500:
                x, y, w, h = cv2.boundingRect(magenta_contour)
                (x,y), radius = cv2.minEnclosingCircle(magenta_contour)
                center = (int(x),int(y))
                radius = int(radius)
                cv2.circle(video,center,radius,(255,0,255),3) # Drawing magenta circles

    # cv2.imshow("Green Mask", green_mask)  # Displaying green mask image
    # cv2.imshow("Magenta Mask", magenta_mask)  # Displaying magenta mask image
    cv2.imshow("Window", video)  # Displaying webcam image

    key = cv2.waitKey(1)
    if key == 27:  # Exit on ESC
        break

cam.release()
cv2.destroyAllWindows()
