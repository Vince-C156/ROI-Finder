import cv2
import torch
from torch import nn
from functions import find_ROI
import os


if __name__ == '__main__':

  """creating cv2 video capture"""
  cap = cv2.VideoCapture('vid2.avi')
  cv2.waitKey(0)

  #frame limit for testing
  frame_limit = 0

  #model init
  Yolomodel = torch.hub.load('ultralytics/yolov5', 'custom', path='models/bestv3.pt')

  """looping video"""

  prev_regions = []

  while(cap.isOpened()):

    frame_limit += 1000
    ret, frame = cap.read()

    #code to find ROI
    regions = find_ROI(frame, True, Yolomodel)

    if len(prev_regions) > 0:
      for region in regions:
        for prev_reg in prev_regions:
          print('checking')
          print(region.rectObj.is_intersect(prev_reg.rectObj))

    prev_regions = regions



  #img_path = 'testimgs/test161008.png'

  #frame = cv2.imread(img_path)

