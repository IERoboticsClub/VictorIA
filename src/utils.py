import cv2
import numpy as np

# Create a window to display the image
cv2.namedWindow("Select Points")


def get_four_points(image):
    """
    Allows the user to select four points on the provided image by clicking with the mouse.

    Args:
    image (numpy.ndarray): The input image on which points will be selected.

    Returns:
    list of tuple: A list of four tuples, each containing (x, y) coordinates of a selected point.
    """
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

    # Close the window
    cv2.destroyAllWindows()

    # Return the list of points
    return points


def sort_coordinates(coord):
    """
    Sorts a list of coordinates by y-coordinate and then by x-coordinate.

    Args:
    coord (list of tuple): The list of coordinates to be sorted.

    Returns:
    list of tuple: The sorted list of coordinates.
    """
    detected_circles = [[val[0], val[1]] for val in coord[0]]
    detected_circles = sorted(detected_circles, key=lambda x: x[1])
    coord2 = [detected_circles[7 * i : 7 * (i + 1)] for i in range(0, 6)]
    coord3 = [sorted(val, key=lambda x: x[0]) for val in coord2]
    coord4 = [val for sublist in coord3 for val in sublist]

    return coord4


def draw_circles_from_video(img):
    """
    Detects and draws circles on an image.

    Args:
    img (numpy.ndarray): The input image on which circles will be detected and drawn.

    Returns:
    tuple: A tuple containing two elements - a list of detected circles and the image with circles drawn.
    """
    # Specifying upper and lower ranges of green color to detect in HSV format
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    # Specifying upper and lower ranges of magenta color to detect in HSV format
    lower_magenta = np.array([136, 87, 111])
    upper_magenta = np.array([180, 255, 255])

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
        detected_circles = sort_coordinates(detected_circles)

        def circles2matrix(circles, img):
            matrix = np.zeros((6, 7), dtype=np.uint8)
            # Loop over each circle
            num = 1
            green = 0
            magenta = 0
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for x, y in circles:
                # Extract the center pixel color in the HSV format
                # circle_hsv = cv2.cvtColor(circle_region, cv2.COLOR_BGR2HSV)
                hsv_pixel = np.mean(
                    hsv_img[y - 20 : y + 20, x - 20 : x + 20], axis=(0, 1)
                )
                # Check if the pixel color is within the green color range
                if (
                    hsv_pixel[0] >= lower_green[0]
                    and hsv_pixel[0] <= upper_green[0]
                    and hsv_pixel[1] >= lower_green[1]
                    and hsv_pixel[1] <= upper_green[1]
                    and hsv_pixel[2] >= lower_green[2]
                    and hsv_pixel[2] <= upper_green[2]
                ):
                    matrix[(num - 1) // 7][(num - 1) % 7] = "1"  # 1 = Green
                    green += 1

                if (
                    hsv_pixel[0] >= lower_magenta[0]
                    and hsv_pixel[0] <= upper_magenta[0]
                    and hsv_pixel[1] >= lower_magenta[1]
                    and hsv_pixel[1] <= upper_magenta[1]
                    and hsv_pixel[2] >= lower_magenta[2]
                    and hsv_pixel[2] <= upper_magenta[2]
                ):
                    matrix[(num - 1) // 7][(num - 1) % 7] = "2"  # 2 = Magenta
                    magenta += 1

                num += 1

            print(matrix, "\n")
            return matrix

        acc = 1
        for pt in detected_circles:
            a, b = pt[0], pt[1]

            # Get the color at the center of the circle
            pixel_value = img[b, a]

            # Convert BGR to HSV color space
            pixel_value_hsv = cv2.cvtColor(np.uint8([[pixel_value]]), cv2.COLOR_BGR2HSV)

            if (pixel_value_hsv >= lower_green).all() and (
                pixel_value_hsv <= upper_green
            ).all():
                border_color = (0, 255, 0)  # Green
                label = "G|"  # 'G' for green
            elif (pixel_value_hsv >= lower_magenta).all() and (
                pixel_value_hsv <= upper_magenta
            ).all():
                border_color = (255, 0, 255)  # Magenta
                label = "M|"  # 'M' for magenta
            else:
                border_color = (0, 0, 0)  # Black
                label = ""  # No label

            # Draw the circumference of the circle with the determined border color
            cv2.circle(img, (a, b), 40, border_color, 2)

            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

            # Draw a number in the center of the circle along with the label ('B' or 'M')
            cv2.putText(
                img,
                f"{label}{acc}",
                (a, b),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            acc += 1

    circles2matrix(detected_circles, img)
    return detected_circles, img


def computer_vision():
    """
    Performs computer vision tasks including point selection, homography transformation, and circle detection on a live camera feed.

    Usage:
    - Run the function to open the camera feed and perform the tasks.
    - Press the 'Esc' key to exit the camera feed.

    """
    # Open camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    acc = 0

    while True:
        check, video = cam.read()
        if acc == 0:
            img = video.copy()
            src_pts = get_four_points(img)
            height = 600
            width = 700
            # Convert to a numpy array
            src_pts = np.array(src_pts)
            dst_pts = np.array(
                [[0, 0], [width, 0], [0, height], [width, height]], dtype="float32"
            )
            # Homography transform

        h, status = cv2.findHomography(src_pts, dst_pts)
        img_out = cv2.warpPerspective(video, h, (width, height))
        detected_circles, _withCircles = draw_circles_from_video(img_out)
        cv2.imshow("With circles", _withCircles)
        acc += 1

        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
