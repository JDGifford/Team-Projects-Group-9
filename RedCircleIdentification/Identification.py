import cv2
import numpy as np
import glob
import os

# Once the path where the images will be stored is decided, we'll substitute it in

path = os.path.dirname(__file__) # currently points to the folder the script is in. Can change it for the final product

colorBounds = [np.array([0,0,180]), np.array([40,40,255])]
cropBuffer = 10    #in pixels, extra pixels to get a little bit around the red circles 

# Main loop for the process, pulls in the folder of images to identify and then performs the identification process
def __main__():
    imageAddresses = [f for f in glob.glob(path + "/images/**")]

    for i in imageAddresses:
        TrimImage(i)
    
    # This is just for debugging, waits until a key press before closing the windows
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
        
    

# Takes an image address as input, identifies the red circle in it, then outputs a new image that contains the red circle to an output folder
def TrimImage(imgAddress):
    img = cv2.imread(imgAddress)
    mask = cv2.inRange(img, colorBounds[0], colorBounds[1])
    
    dimensions = img.shape # Height, Width, # of Channels
    
    circles = cv2.HoughCircles(image=mask, method=cv2.HOUGH_GRADIENT, dp=0.9, 
                            minDist=50, param1=110, param2=25, maxRadius=100)
    
    for co, i in enumerate(circles[0, :], start=1):
        cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
        cv2.circle(img,(int(i[0]),int(i[1])),2,(255,0,0),3)
    
    # Still need to figure out some effective bounds checking
    left = int(circles[0][0][0] - circles[0][0][2]) - cropBuffer
    right = int(circles[0][0][0] + circles[0][0][2]) + cropBuffer
    top = int(circles[0][0][1] - circles[0][0][2]) - cropBuffer
    bottom = int(circles[0][0][1] + circles[0][0][2]) + cropBuffer
    
    cropped_img = img[top:bottom, left:right]
    
    #For debug purposes, shows the mask in a window
    cv2.namedWindow(imgAddress, cv2.WINDOW_NORMAL)
    cv2.imshow(imgAddress, cropped_img)
    
__main__()