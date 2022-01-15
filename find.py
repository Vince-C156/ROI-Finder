"""
Using ORB feature detection to crop ROI with target
"""

#uninstall opencv and install opencv contribs
#downgrading opencv contrib to version 3.4.11

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import argparse
from classes import ROI


"""creating ORB object to process features on image with .detectAndCompute"""

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img_path", required=True, help="Path to image")
args = vars(ap.parse_args())

#gray image copied to draw features on for more visiblity of features
img = cv2.imread(args['img_path'])
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#using canny edge detection with a high threshold to filter out background noise(ie trees)
img = cv2.Canny(img,400,600)

#orb object created to process image keypoints and descriptors
orb = cv2.ORB_create(500)
keypoints_sift, descriptors = orb.detectAndCompute(img, None)

#drawing keypoints and descriptors on canny edge processed image
img_processed = cv2.drawKeypoints(img, keypoints_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_processed)

"""Using thresholding and blur to get blobs for ROI"""

gray_processed = cv2.cvtColor(img_processed,cv2.COLOR_BGR2GRAY)

#applying median blur to filter out smaller and less dense feature groups
median_blur = cv2.medianBlur(gray_processed,7)

#applying gaussian blur to make dense feature groups more blob like
gaussian_blur = cv2.GaussianBlur(median_blur,(21,21),20)
plt.imshow(gaussian_blur)

"""Finding contours of blurs.

Then finding center of those countours to create a bounding box around the ROI
"""

_, threshold = cv2.threshold(gaussian_blur,40,200,cv2.THRESH_BINARY)
contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[1:-1]

#list for ROI objects
ROIS = []

for cnt in contours:

  for c in cnt:
    M = cv2.moments(c)

    if 0 not in M.values():
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
    else:
      cX = 0
      cY = 0

    if cv2.contourArea(c) > 300:
      ROIS.append(ROI(len(ROIS), [cX, cY]))
      

    cv2.circle(threshold, (cX, cY), 7, (255, 255, 255), -1)

  cv2.drawContours(threshold, cnt, -1, (255,0,0), 5)


#size of ROI (rectangle)
height, width = 300, 300

print("DETECTED CENTER POINTS OF CONTOURS")
print("-----------------------------------")
for ROI in ROIS:
    print(ROI.original_center)
print("-----------------------------------")

height_div2, width_div2 = height // 2, width // 2

#storing bounding rectangle top left corner and bottom right corner coordinates
rectangle_list = []


#using centers of countours to draw bounding box
for ROI in ROIS:
  x, y = ROI.original_center[0], ROI.original_center[1]
  top_leftx, top_lefty = x - width_div2, y + height_div2
  bottom_rightx, bottom_righty = x + width_div2, y - height_div2
  cv2.rectangle(threshold,(top_leftx, top_lefty),(bottom_rightx, bottom_righty),(255,255,255),5)
  ROI.add_shape([top_leftx, top_lefty, bottom_rightx, bottom_righty])
  rectangle_list.append([top_leftx, top_lefty, bottom_rightx, bottom_righty])


plt.imshow(threshold)

"""Crop out ROI's"""


img = cv2.imread(args['img_path'])

for ROI in ROIS:
  rect = ROI.shape
  rect = list(map(lambda x: max(x,0),rect))
  x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
  cropped = img[y2:y2 + 200, x1:x2]
  ROI.add_img(cropped)

"""Displaying ROI's"""

fig = plt.figure(figsize=(10, 10))

print(f'REGIONS DETECTED: {len(ROIS)}')
rows = 1
columns = len(ROIS)

for i, ROI in enumerate(ROIS):
  fig.add_subplot(rows, columns, i + 1)
  plt.imshow(ROI.img)
plt.show()

plt.imshow(img)

"""Testing shape detect with yolov5 model"""



Yolomodel = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestv3.pt')

img = cv2.imread(args['img_path'])


for idx, ROI in enumerate(ROIS):
    rescaled = cv2.resize(ROI.img, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
    blue, green, red = cv2.split(rescaled)
    #rescaled = cv2.merge((red, green, blue))
    results = Yolomodel(rescaled)
    results.render()
    results.print()
    cv2.imshow('result', rescaled)
    cv2.waitKey(0)
    ROI.add_bb(results.xyxy)



"""Finding Center of bounding box"""

for ROI in ROIS:
    BB_list = ROI.bb_rectangles[ROI.idx][0]

    for tensor in BB_list:
        if tensor.shape[0] > 0:
            BB = tensor
            x1, y1, x2, y2 = int(BB[0])/6 , int(BB[1])/6 , int(BB[2])/6 , int(BB[3])/6
            detection = BB[5]
            xCenter = int((x1 + x2) / 2)
            yCenter = int((y1 + y2) / 2)

            ROI.add_obj_centers(detection, xCenter, yCenter)
                            
            cv2.circle(ROI.img, (xCenter, yCenter), 3, (255, 255, 255), -1)

            print('---------------------------------------')
            print('CENTER OF DETECTIONS [class, x, y]')
            print(ROI.obj_centers)
            print('---------------------------------------')



