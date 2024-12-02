import paramiko.client
import cv2
import numpy as np
import glob
import os
import datetime

from paramiko import SSHClient

# Once the path where the images will be stored is decided, we'll substitute it in

path = os.path.dirname(__file__) # currently points to the folder the script is in. Can change it for the final product

colorBounds = [np.array([0,0,220]), np.array([50,50,255])]
cropBuffer = 7    #in pixels, extra pixels to get a little bit around the red circles 

# Main loop for the process, pulls in the folder of images to identify and then performs the identification process
def __main__():
    logFile = open(path + "/logs/log.txt", "a")
    imageAddresses = [f for f in glob.glob(path + "/images/**")]
    imgId = 0
    
    if len(imageAddresses) > 0:
        for i in imageAddresses:
            TrimImage(i, imgId, logFile)
            imgId = imgId + 1

        uploadImages()
    else:
        logFile.write("No images found in directory.")
    logFile.close()


# Takes an image address as input, identifies the red circle in it, then outputs a new image that contains the red circle to an output folder
def TrimImage(imgAddress, id, log):
    img = cv2.imread(imgAddress)
    mask = cv2.inRange(img, colorBounds[0], colorBounds[1])
    
    dimensions = img.shape # Height, Width, # of Channels
    
    circles = cv2.HoughCircles(image=mask, method=cv2.HOUGH_GRADIENT_ALT, dp=1, 
                            minDist=50, param1=300, param2=0.7)

    if (circles is not None): # Will skip an image if no circles are found
        
        # Debug code for displaying where circles were identified
        #for co, i in enumerate(circles[0, :], start=1):
        #    cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
        #    cv2.circle(img,(int(i[0]),int(i[1])),2,(255,0,0),3)
        
        left = max(int(circles[0][0][0] - circles[0][0][2]) - cropBuffer, 0)
        right = min(int(circles[0][0][0] + circles[0][0][2]) + cropBuffer, dimensions[1])
        top = max(int(circles[0][0][1] - circles[0][0][2]) - cropBuffer, 0)
        bottom = min(int(circles[0][0][1] + circles[0][0][2]) + cropBuffer, dimensions[0])

        cropped_img = img[top:bottom, left:right]
        
        cv2.imwrite(path + "/output/" + str(id) + "_cropped.png", cropped_img)
    else:
        log.write(str(datetime.datetime.now()) + ": Could not find circles in: " + imgAddress + "\n")

def uploadImages():
    images = [f for f in glob.glob(path + "/output/**")]

    client = SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect('3.213.86.237', username='dse-user', key_filename='./dse-user.pem', timeout=30)
    sftp_client = client.open_sftp()

    i = 0
    for image in images:
        i += 1
        remote_path = '/home/dse-user/MobileApp/public/image/ds{:02d}.jpg'.format(i) 
        sftp_client.put(image, remote_path)

    sftp_client.close()
    client.close()


__main__()
