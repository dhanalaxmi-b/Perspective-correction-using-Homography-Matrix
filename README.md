
Homography transformation:

  In the field of computer vision, any two images of the same planar surface in space are related by a homography.Computing a matrix which transforms a quadrangle to another quadrangle in 2D is known as Homography matrix. This has many practical applications, such as image rectification, image registration, or computation of camera motion—rotation and translation—between two images

Using Homography matrix ,we are going to correct perspective distortion in image.


#Perspective-Correction-using-Homography

Step 1: Estimate the corners coordiantes of object in both the image

Step 2: Determine homography matrix 

Step 3: Change the image perspective based on the homography matrix of object

