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


def color_quantize(img, K):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    quantized_img = res.reshape((img.shape))

    label_img = label.reshape((img.shape[:2]))
    return label_img, quantized_img


def find_ROI(frame, visualize, Yolomodel):

  """
  This function takes in a frame and outputs the pixel coordinates and boundingbox
  crop of objected detected
  
  input: frame from video (arbitrary dimension), True or False to visualize images

  output: tuple with ROI crop and pixel coordinates relative to orginal images [(detection, img, [x,y]), (detection, img, [x,y]), ...]
  """

  from classes import ROI
  from classes import rectangle

  """bounding box space to determin when ROI focuses"""
  bb_featureSpace = np.ones(frame.shape,np.uint8) * 255


  """creating ORB object to process features on image with .detectAndCompute"""


  #using canny edge detection with a high threshold to filter out background noise(ie trees)
  #creating copy to processes on to preserve orginal frame
  cv2.imshow('original', frame)
  img = frame
  img = cv2.Canny(img,400,600, L2gradient = True)

  #orb object created to process image keypoints and descriptors
  orb = cv2.ORB_create(200)
  keypoints_sift, descriptors = orb.detectAndCompute(img, None)

  #drawing keypoints and descriptors on canny edge processed image
  img_processed = cv2.drawKeypoints(img, keypoints_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  """Using thresholding and blur to get blobs for ROI"""

  gray_processed = cv2.cvtColor(img_processed,cv2.COLOR_BGR2GRAY)

  #applying median blur to filter out smaller and less dense feature groups
  median_blur = cv2.medianBlur(gray_processed,7)

  #applying gaussian blur to make dense feature groups more blob like
  gaussian_blur = cv2.GaussianBlur(median_blur,(21,21),20)

  #threshing blur
  _, threshold = cv2.threshold(gaussian_blur,40,200,cv2.THRESH_BINARY)

  """Finding contours of blurs.

  Then finding center of those countours to create a bounding box around the ROI
  """

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
  height, width = 100, 100

  height_div2, width_div2 = height // 2, width // 2

  #using centers of countours to draw bounding box
  for ROI in ROIS:
    
    x, y = ROI.original_center[0], ROI.original_center[1]
    
    top_rightx, top_righty = x + width_div2, y - height_div2
    bottom_leftx, bottom_lefty = x - width_div2, y + height_div2

    bb_rect = rectangle(bottom_leftx, bottom_lefty, top_rightx, top_righty)

    ROI.add_rectObj(bb_rect)

    bottom_leftptn = (bottom_leftx, bottom_lefty)
    top_rightptn = (top_rightx, top_righty)
    
    if visualize == True:
      cv2.rectangle(frame, bottom_leftptn,top_rightptn,(128,128,128),5)
      cv2.rectangle(bb_featureSpace, bottom_leftptn ,top_rightptn,(128,128,128),5)
      

  if visualize == True:
    cv2.imshow('bbframe', frame)
    cv2.imshow('bbfeatures', bb_featureSpace)
    cv2.waitKey(2)

  return ROIS




      


      
      



