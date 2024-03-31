import cv2
import numpy as np

def signal(img, bboxes):
    
    #crop image about traffic light boundig box
    cropped_img = img[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2])]
    cv2.imshow("Cropped Image", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #convert to hsv
    hsv_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for red, yellow, and green in HSV
    lower_red = np.array([90, 100, 100])
    upper_red = np.array([110, 255, 255])

    lower_yellow = np.array([120, 100, 100])
    upper_yellow = np.array([140, 255, 255])

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([70, 255, 255])
    # Create masks for each color
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    # Determine the dominant color based on the percentage of pixels
    max_pixels = max(red_pixels, green_pixels)
    if max_pixels == red_pixels:
        return "red"
    elif max_pixels == green_pixels:
        return "green"
    else:
        return "yellow"
    