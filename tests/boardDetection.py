import cv2
import numpy as np


# get the 4 corner manually
def get_four_points(image):
    # Create a window to display the image
    cv2.namedWindow("Select Points")

    # Create an empty list to store the points
    points = []

    # Define the callback function for mouse events
    def mouse_callback(event, x, y, flags, param):
        # If the left mouse button is clicked, add the current point to the list
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw a circle at the current point
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            # Update the display
            cv2.imshow("Select Points", image)

    # Display the image and wait for mouse clicks
    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", mouse_callback)
    cv2.waitKey(0)

    # Check if exactly 4 points were selected
    if len(points) != 4:
        print("Error: Please select exactly 4 points")
        return None

    # close the window
    cv2.destroyAllWindows()

    # Return the list of points
    return points


def sort_coordinates(coord):
    detected_circles = [[val[0], val[1]] for val in coord[0]]
    detected_circles = sorted(detected_circles, key=lambda x: x[1])
    coord2 = [detected_circles[7 * i : 7 * (i + 1)] for i in range(0, 6)]
    coord3 = [sorted(val, key=lambda x: x[0]) for val in coord2]
    coord4 = [val for sublist in coord3 for val in sublist]

    return coord4


def draw_circles_from_video(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        1,
        92,
        param1=82,
        param2=25,
        minRadius=10,
        maxRadius=55,
    )

    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        # detected_circles= [ [val[0], val[1]] for val in detected_circles[0]]
        detected_circles = sort_coordinates(detected_circles)

        acc = 1
        for pt in detected_circles:
            a, b = pt[0], pt[1]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), 40, (0, 255, 0), 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            if acc < 11:
                cv2.putText(
                    img,
                    str(acc),
                    (a, b),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif acc < 21:
                cv2.putText(
                    
                    img,
                    str(acc),
                    (a, b),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            elif acc < 31:
                cv2.putText(
                    img,
                    str(acc),
                    (a, b),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 30),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    img,
                    str(acc),
                    (a, b),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 60, 0),
                    2,
                    cv2.LINE_AA,
                )
            # draw a number in the center of the circle
            # cv2.putText(img, str(acc), (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            acc += 1
    return detected_circles, img


# open camera
cam = cv2.VideoCapture(0)
acc = 0

while True:
    check, frame = cam.read()
    if acc == 0:
        img = frame.copy()
        src_pts = get_four_points(img)
        height = 600
        width = 700
        # p to a numpy array
        src_pts = np.array(src_pts)
        dst_pts = np.array(
            [[0, 0], [width, 0], [0, height], [width, height]], dtype="float32"
        )
        # homography transform

    h, status = cv2.findHomography(src_pts, dst_pts)
    img_out = cv2.warpPerspective(frame, h, (width, height))

    detected_circles, _withCircles = draw_circles_from_video(img_out)
    cv2.imshow("video", frame)
    cv2.imshow("With circles", _withCircles)
    acc += 1

    key = cv2.waitKey(1)
    if key == 27:
        break
