import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("eldenring.png")

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cimg = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#

lowerRed = np.array([10,10,180])
upperRed = np.array([40,40,255])

mask = cv2.inRange(img, lowerRed, upperRed)
res = cv2.bitwise_and(img, img, mask=mask)

#mask = cv2.medianBlur(mask, 5)

#circles = cv2.HoughCircles(image=mask, method=cv2.HOUGH_GRADIENT, dp=0.9, 
#                            minDist=10, param1=110, param2=35, maxRadius=100)

circles = cv2.HoughCircles(image=mask, method=cv2.HOUGH_GRADIENT, dp=0.9, 
                            minDist=50, param1=110, param2=20, maxRadius=100)

#print(circles)

#cv2.circle(cimg, (int(circles[0][0][0]), int(circles[0][0][1])), int(circles[0][0][2])+5, (0,255,0),2)
#cv2.circle(cimg, (int(circles[0][0][0]), int(circles[0][0][1])), 2, (0,0,255),2)

for co, i in enumerate(circles[0, :], start=1):
    cv2.circle(cimg,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
    cv2.circle(cimg,(int(i[0]),int(i[1])),2,(0,0,255),3)
#print("Number of circles detected: ", co)

cv2.arrowedLine(cimg, (0,0), (int(circles[0][0][0]), int(circles[0][0][1])), (255,255,255), 3)

cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

cv2.imshow("mask", mask)
cv2.imshow("img", cimg)

if cv2.waitKey(0):
    cv2.destroyAllWindows()



#print(circles.size)
#print(circles[0][0][0])
#cv2.circle(cimg, (int(circles[0][0][0]), int(circles[0][0][1])), int(circles[0][0][2]), (0,255,0),2)

#for co, i in enumerate(circles[0, :], start=1):
#    cv2.circle(cimg,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
#    cv2.circle(cimg,(int(i[0]),int(i[1])),2,(255,0,0),3)
#print("Number of circles detected: ", co)


#plt.imshow(img)
#plt.show()