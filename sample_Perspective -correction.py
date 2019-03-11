import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from cv2 import CAP_IMAGES
 
#Load the images
image = cv2.imread("/home/dhanalaxmi/workspace/AI/Height_detection/image/object/hg.jpeg")
im=cv2.imread("/home/dhanalaxmi/workspace/AI/Height_detection/image/object/hg1.jpeg")
#Corner coordinates
image_corner = np.float32([[81,447], [532,447], [94,756],[531,758]])
cv2.circle(image, (81,447), 8, (255, 0, 0), 5)
cv2.circle(image, (532,447), 8, (255, 0, 0), 5)
cv2.circle(image, (94,756), 8, (255, 0, 0), 5)
cv2.circle(image, (531,758), 8, (255, 0, 0), 5)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()

im_corner = np.float32([[43,517], [288,480], [77,718], [322,703]])
cv2.circle(im, (43,517), 8, (255, 0, 0), 5)
cv2.circle(im, (288,480), 8, (255, 0, 0), 5)
cv2.circle(im, (77,718), 8, (255, 0, 0), 5)
cv2.circle(im, (322,703), 8, (255, 0, 0), 5)

plt.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
plt.show()

#Homography matrix
h, mask = cv2.findHomography(im_corner,image_corner)
  
out = cv2.warpPerspective(im, h, (1000,1000))

#Shoe the image
plt.imshow(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
plt.show()
