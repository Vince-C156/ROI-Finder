import cv2
import torch
from torch import nn
from functions import find_ROI
import os


if __name__ == '__main__':

  debug = True

  """creating cv2 video capture"""
  cap = cv2.VideoCapture('vid2.avi')
  cv2.waitKey(0)

  #frame limit for testing
  frame_count = 0

  #variables used to keep track of detected ROI's
  duplicate = False
  prev_regions = list()
  ROI_descriptors = list()

  #methods for image matching to mimimize duplicates
  orb = cv2.ORB_create(50)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)



  """looping video"""

  while(cap.isOpened()):

    frame_count += 1
    ret, frame = cap.read()

    #code to find ROI
    regions = find_ROI(frame, True)

    #filtering out frames with too many ROI's

    if ( (len(prev_regions) > 0) & (len(regions) < 5) ):

      #checking every region with every previous region to check for any region intersection
      for region in regions:

        for prev_reg in prev_regions:

          intersecting = region.rectObj.is_intersect(prev_reg.rectObj)

          if (intersecting):

              #calculating intersection over union of both bounding boxes

              IoU = region.rectObj.IoU(prev_reg.rectObj)

              if debug == True:
                  IoUobj = region.rectObj & prev_reg.rectObj
                  Lptn = IoUobj.bounding_box[0], IoUobj.bounding_box[1]
                  Rptn = IoUobj.bounding_box[2], IoUobj.bounding_box[3]
                  cv2.rectangle(frame, Lptn, Rptn,(0,0,255),5)
                  cv2.imshow('intersections', frame)
                  cv2.waitKey(1)
              
              #counting relevant ROI's as regions that have an IoU of 0.8 or higher over two frames
              if IoU >= 0.8:
                  x1, y1, x2, y2 = region.rectObj.bounding_box[0], region.rectObj.bounding_box[1], region.rectObj.bounding_box[2], region.rectObj.bounding_box[3]

                  focused_region = frame[y2:y1, x1:x2]

                  #finding features of region to filter out duplicates
                  #then iterating through features of found ROI's to check for matches

                  kp, des1 = orb.detectAndCompute(focused_region, None)
                  for des2 in ROI_descriptors:
                      num_matches = len(bf.match(des1, des2))
                      if num_matches >= 10:
                          duplicate = True
                          break
                  if duplicate == True:
                      print("DUPLICATE WITH ", num_matches, "MATCHES")
                      duplicate = False
                      break
                      
                  #if not a duplicate append the descriptors to the list of found ROI descriptors
                  ROI_descriptors.append(des1)
                  
                  #writing file then breaking out of loop
                  filename = 'ROIS/' + str(frame_count) + '.jpg'
                  cv2.imwrite(filename, focused_region)
                  break

    prev_regions = regions



