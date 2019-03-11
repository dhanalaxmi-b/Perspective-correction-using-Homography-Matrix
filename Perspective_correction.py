'''
Created on 11-Mar-2019

@author: dhanalaxmi
'''
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
from imutils import perspective

#load an image
image = cv2.imread("/home/dhanalaxmi/workspace/AI/Height_detection/image/object/hg.jpeg")
im=cv2.imread("/home/dhanalaxmi/workspace/AI/Height_detection/image/object/hg1.jpeg")

#Find corner co-ordinates with straight image
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
thresh = cv2.threshold(gray, 100,220, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
  
cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
count=0
for c in cnts: 
    area=cv2.contourArea(c)
    if area>120000 and area<180000 :
        epsilon = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.0099* epsilon, True)
        if len(approx)== 4 :
            box = cv2.minAreaRect(approx)
            box_angle=box[-1]
            box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
            for (x, y) in box:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), 3)
 
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()
 
object_corner=box

#find the corner co-ordinates with tilted object in image 
gray = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
thresh = cv2.threshold(gray, 100,220, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
      
cnts = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
count=0
for c in cnts: 
    area=cv2.contourArea(c)
    if area>50000 and area<60000 :
        epsilon = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.099* epsilon, True)
        if len(approx)==4:
            # determine the most extreme points along the contour
            tl = tuple(c[c[:, :, 0].argmin()][0])
            br = tuple(c[c[:, :, 0].argmax()][0])
            tr = tuple(c[c[:, :, 1].argmin()][0])
            bl = tuple(c[c[:, :, 1].argmax()][0])
            cv2.circle(im, tl, 8, (0, 0, 255), -1)
            cv2.circle(im, br, 8, (0, 255, 0), -1)
            cv2.circle(im, tr, 8, (255, 0, 0), -1)
            cv2.circle(im, bl, 8, (255, 255, 0), -1)
                
plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
plt.show()
 
obj_corner=(tl,tr,br,bl)
obj_corner=(np.asarray(obj_corner, dtype="float32"))

#Estimate Homography matrix

h, mask = cv2.findHomography(obj_corner,object_corner)
      
out = cv2.warpPerspective(im, h, (1500,1000))
plt.imshow(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
plt.show()


