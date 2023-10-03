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
webcam_video = cv2.VideoCapture(0)

while True:
    success, video = webcam_video.read()  # Reading webcam footage

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
                cv2.rectangle(
                    video, (x, y), (x + w, y + h), (0, 255, 0), 3
                )  # Drawing green rectangles

    # Finding position of all magenta contours
    if len(magenta_contours) != 0:
        for magenta_contour in magenta_contours:
            if cv2.contourArea(magenta_contour) > 500:
                x, y, w, h = cv2.boundingRect(magenta_contour)
                cv2.rectangle(
                    video, (x, y), (x + w, y + h), (255, 0, 255), 3
                )  # Drawing magenta rectangles

    cv2.imshow("Green Mask", green_mask)  # Displaying green mask image
    cv2.imshow("Magenta Mask", magenta_mask)  # Displaying magenta mask image
    cv2.imshow("Window", video)  # Displaying webcam image

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # Exit the loop when 'q' is pressed

webcam_video.release()
cv2.destroyAllWindows()
