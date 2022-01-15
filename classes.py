import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import argparse

"""Code to download local images to test on"""

"""Class for rectangles"""

#if IoU > .9 write ROI

class rectangle:

    bounding_box = list()

    def __init__(self, bottom_leftx, bottom_lefty, top_rightx, top_righty):
        self.bounding_box = [bottom_leftx, bottom_lefty, top_rightx, top_righty]

    def __or__(self, other_rect):
        pass

    def __and__(self, other_rect):

        print('passed')

        debug = True

        Rx = min(self.bounding_box[2], other_rect.bounding_box[2])
        Ry = max(self.bounding_box[3], other_rect.bounding_box[3])

        #100 , 100

        #width of each rectangle respectively
        width_1 = self.bounding_box[2] - self.bounding_box[0]
        width_2 = other_rect.bounding_box[2] - other_rect.bounding_box[0]

        #height of each rectangle respectively
        height_1 = self.bounding_box[1] - self.bounding_box[3]
        height_2 = other_rect.bounding_box[1] - other_rect.bounding_box[3]

        w = max(self.bounding_box[0] + width_1, other_rect.bounding_box[0] + width_2) - Rx
        h = min(self.bounding_box[3] + height_1, other_rect.bounding_box[3] + height_2) - Ry

        if debug:
            print(f"Rx :{Rx} Ry:{Ry}")
            print("===================")
            print(f"width_1 :{width_1} width_2:{width_2}")
            print("===================")
            print(f"height_1:{height_1} height_2:{height_2}")
            print("===================")
            print(w, h)

        if ( (w > 0) & (h > 0) ):
            return [Rx, Ry, (Rx - w), (Ry + h)]
        else:
            return 0

    @staticmethod
    def checkin_range(bb_start, min_val, max_val):

        if ( (bb_start >= min_val) and (bb_start <= max_val) ):
            return True
        else:
            return False
        

    def in_range(self, other_rect, self_greater, axis = 'X'):

        if axis == 'X':

            if self_greater:

                return self.checkin_range(self.bounding_box[0], other_rect.bounding_box[0], other_rect.bounding_box[2])

            else:

                return self.checkin_range(other_rect.bounding_box[0], self.bounding_box[0], self.bounding_box[2])
                
        else:

            if self_greater:

                return self.checkin_range(self.bounding_box[3], other_rect.bounding_box[3], other_rect.bounding_box[1])

            else:

                return self.checkin_range(other_rect.bounding_box[3], self.bounding_box[3], self.bounding_box[1])
            

    def is_intersect(self, other_rect):

        debug = False

        x_inRange = False
        y_inRange = False

        """checking if x values intersect between both bounding boxes"""

        if (self.bounding_box[0] > other_rect.bounding_box[0]):

            x_inRange = self.in_range(other_rect, True, axis = 'X')
            
        elif(self.bounding_box[0] < other_rect.bounding_box[0]):

            x_inRange = self.in_range(other_rect, False, axis = 'X')
            
        elif(self.bounding_box[0] == other_rect.bounding_box[0]):
            x_inRange = True

        else:
            return False


        if debug:
            if x_inRange:
                print('X IN RANGE')
            else:
                print('X NOT IN RANGE')


        """checking if y values intersect between both bounding boxes"""

        if debug:
            print("CHECKING Y")

        if (self.bounding_box[3] > other_rect.bounding_box[3]):

            y_inRange = self.in_range(other_rect, True, axis = 'Y')
            
        elif(self.bounding_box[3] < other_rect.bounding_box[3]):

            y_inRange = self.in_range(other_rect, False, axis = 'Y')
            
        elif(self.bounding_box[3] == other_rect.bounding_box[3]):
            y_inRange = True

        else:
            return False

        if debug:
            print(f"VALUE OF Y IS {y_inRange}")


        if (x_inRange & y_inRange):
            return True
        else:
            return False
  

        

"""Class for ROI data"""

"""
    def add_obj_centers(self, detection, x, y):
        detection_center = [x,y]
        print('DETECTION CLASS')
        print(detection)
        self.obj_centers.update({detection: detection_center})

    def add_img(self, img):
        self.img = img

    def add_shape(self, shape):
        self.shape = shape
"""

class ROI:

    rel_center = []
    bb_rectangles = []

    def __init__(self, idx, original_center):
        self.idx = idx
        print(original_center)
        self.original_center = original_center

    def update_idx(self, idx):
        if idx < 0:
            idx = 0
        self.idx = idx

    def replace_original_center(self, x, y):
        self.original_center = [x, y]

    def add_bb(self, bb):
        self.bb_rectangles.append(bb)

    def add_detections_cropped(self, detection, img, x, y):
        self.detections_cropped.append((detection, img, [x,y]))

    def add_rectObj(self, rect):
        self.rectObj = rect
