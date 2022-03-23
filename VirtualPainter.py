# Importing the libraries
import cv2
import numpy as np
import time 
import os
import HandTrackingModule as htm


# Global variables
brush_thickness = 15
eraser_thickness = 50


# Reading the header images
folder_path = "header"
my_list = os.listdir(folder_path)

if ".DS_Store" in my_list:
    os.system("rm header/.DS_Store")
    my_list.remove(".DS_Store")

overlay_list = [cv2.imread(f"{folder_path}/{image_path}") for image_path in my_list]

# Initialising the header & the camera dimensions
header = overlay_list[0]
text = "Pink"
colour = (255, 0, 255)

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(detectionCon=0.85)
xp, yp = 0, 0
canvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    # Read the image
    success, img = cap.read()
    img = cv2.flip(img, 1) 

    # Find hand landmarks
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw = False)

    if len(lmList) != 0:
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingers_up()

        # Selection mode when two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), colour, cv2.FILLED)

            # Checking for menu selection
            if y1 < 117:
                if 250 < x1 < 450:
                    header = overlay_list[0] # Pink
                    text = "Pink"
                    colour = (255, 0, 255)

                elif 550 < x1 < 750:
                    header = overlay_list[1] # Blue
                    text = "Blue"
                    colour = (255, 0, 0)

                elif 1050 < x1 < 1200:
                    header = overlay_list[2] # Eraser
                    text = "Eraser"
                    colour = (0, 0, 0)

                elif 800 < x1 < 950:
                    header = overlay_list[3] # Green
                    text = "Green"
                    colour = (0, 255, 0)

        # Drawing mode when index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, colour, cv2.FILLED)

            # Starting the line at the current point
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if colour == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), colour, eraser_thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), colour, eraser_thickness)
            
            else:
                cv2.line(img, (xp, yp), (x1, y1), colour, brush_thickness)
                cv2.line(canvas, (xp, yp), (x1, y1), colour, brush_thickness)

            xp, yp = x1, y1

    # Merging the camera and the canvas
    img_grey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inverse = cv2.threshold(img_grey, 50, 255, cv2.THRESH_BINARY_INV)
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, canvas)

    # Placement of the header on the frame
    img[0:117, 0:1280] = header 
    cv2.putText(img, text, (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1, colour, 2)

    # Putting the canvas on the video
    img = cv2.addWeighted(img, 0.8, canvas, 0.8, 0)

    cv2.imshow("Image", img)

    # Closing the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 


cap.release()    
cv2.destroyAllWindows()
