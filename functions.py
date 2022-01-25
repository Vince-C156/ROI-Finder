"""
Using ORB feature detection to crop ROI with target
"""

#uninstall opencv and install opencv contribs
#downgrading opencv contrib to version 3.4.11

import cv2
import numpy as np


def find_ROI(frame, visualize):

  """
  This function takes in a frame and outputs a list of ROI objects which contain information about the region
  such as center and crop coordinates
  
  input: frame from video (arbitrary dimension), True or False to visualize images

  output: list of ROI objects
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
  img = cv2.Canny(img,300, 600, L2gradient = True)

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
        region = ROI(len(ROIS), [cX, cY])
        
        minRect = cv2.minAreaRect(c)
        box = np.int0( cv2.boxPoints(minRect) )
        
        xmin, ymin = min(box[0:-1, 0]), min(box[0:-1, 1])
        xmax, ymax = max(box[0:-1, 0]), max(box[0:-1, 1])        


        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 256, 0), 1)
        print(f"minimum {xmin} {ymin}")
        print(f"maxmium {xmax} {ymax}")

        cv2.drawContours(frame, [box], 0, color=(128, 128, 128), thickness=1)        
        rect = rectangle(xmin, ymin, xmax, ymax)

        region.add_rectObj(rect)
        ROIS.append(region)
        
      cv2.circle(threshold, (cX, cY), 7, (255, 255, 255), -1)

    cv2.drawContours(threshold, cnt, -1, (255,0,0), 5)


  #size of ROI (rectangle)
  height, width = 200, 200

  height_div2, width_div2 = height // 2, width // 2
  """
  #using centers of countours to draw bounding box
  for ROI in ROIS:
    

    x, y = ROI.original_center[0], ROI.original_center[1]   
    top_rightx, top_righty = max(0,x + width_div2), max(0,y - height_div2)
    bottom_leftx, bottom_lefty = max(0,x - width_div2), max(0, y + height_div2)

    #top_rightx, top_righty, bottom_leftx, bottom_lefty =
    bb_rect = rectangle(bottom_leftx, bottom_lefty, top_rightx, top_righty)

    ROI.add_rectObj(bb_rect)

    bottom_leftptn = (bottom_leftx, bottom_lefty)
    top_rightptn = (top_rightx, top_righty)
    
    if visualize == True:
      cv2.rectangle(frame, bottom_leftptn,top_rightptn,(128,128,128),5)
      cv2.rectangle(bb_featureSpace, bottom_leftptn ,top_rightptn,(128,128,128),5)
  """
 

  if visualize == True:
    cv2.imshow('bbframe', frame)
    cv2.imshow('bbfeatures', bb_featureSpace)
    cv2.waitKey(1)

  return ROIS




      


      
      



