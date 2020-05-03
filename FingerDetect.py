# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 09:56:07 2020

@author: Bombita & See
"""

from __future__ import print_function

import numpy as np
import cv2 as cv

from video import create_capture
from PIL import ImageGrab

fourcc = cv.VideoWriter_fourcc(*'XVID')
screencap = cv.VideoWriter('ScreenCap.avi', fourcc, 15, (1920,1080))

total = 9
R_rect1 = None
C_rect1 = None
R_rect2 = None
C_rect2 = None
data = None
traverse = []
draw = []
n = 1

def contour(mask):
    grayed = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(grayed,0,255,0)
    conts, hier = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return conts

#Rectangle drawer to indicate region
def scan_region(scene):
    row, column, _ = scene.shape
    global total, R_rect1, C_rect1, R_rect2, C_rect2
    
    R_rect1 = np.array([6 * row / 20, 6 * row / 20, 6 * row / 20, 9 * row / 20, 9 * row / 20, 9 * row / 20, 12 * row / 20,12 * row / 20, 12 * row / 20], dtype=np.uint32)
    C_rect1 = np.array([9 * column / 20, 10 * column / 20, 11 * column / 20, 9 * column / 20, 10 * column / 20, 11 * column / 20, 9 * column / 20, 10 * column / 20, 11 * column / 20], dtype=np.uint32)
    R_rect2 = R_rect1 + 10
    C_rect2 = C_rect1 + 10
    
    for i in range(total):
        cv.rectangle(scene, (C_rect1[i], R_rect1[i]), (C_rect2[i], R_rect2[i]), (0,0,0), 1)
    return scene



#Gets the pixels of hand and make HSV
def hand_data(scene):
    global R_rect1, C_rect2
    
    HSV_scene = cv.cvtColor(scene, cv.COLOR_BGR2HSV)
    region = np.zeros([90,10,3], dtype=HSV_scene.dtype)
    
    for i in range(total):
        region[i * 10: i * 10 + 10, 0:10] = HSV_scene[R_rect1[i]:R_rect1[i]+10, C_rect1[i]:C_rect1[i]+10]
        
    data = cv.calcHist([region],[0,1], None, [180,256], [0,180,0,256])
    return cv.normalize(data, data, 0, 255, cv.NORM_MINMAX)

#Removing of background from the hand
def histogram_maker(scene, histo):
    HSV = cv.cvtColor(scene, cv.COLOR_BGR2HSV)
    hsvED = cv.calcBackProject([HSV], [0,1], histo, [0,180,0,256], 1)
    
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31,31))
    cv.filter2D(hsvED, -1, disc, hsvED)
    
    ret, thresh = cv.threshold(hsvED, 150, 255, cv.THRESH_BINARY)
    thresh = cv.merge((thresh,thresh,thresh))
    return cv.bitwise_and(scene, thresh)

#Contour points of the hand
def maximum(contour_list):
    MAX_i = 0
    MAX_area = 0
    
    for i in range(len(contour_list)):
        cont = contour_list[i]
        
        cont_area = cv.contourArea(cont)
        
        if cont_area > MAX_area:
            MAX_i = i
            MAX_area = cont_area
        return contour_list[MAX_i]

#Getting the farthest defect from the center of the hand
def farthest(defect, cnt, contour_centroid):
    if defect is not None and contour_centroid is not None:
        z = defect[:, 0][:, 0]
        cx, cy = contour_centroid
        
        x = np.array(cnt[z][:, 0][:, 0], dtype=np.float)
        y = np.array(cnt[z][:, 0][:, 1], dtype=np.float)
        xp = cv.pow(cv.subtract(x, cx), 2)
        yp = cv.pow(cv.subtract(y, cy), 2)
        distance = cv.sqrt(cv.add(xp, yp))
        MAX_distanceI = np.argmax(distance)
        
        if MAX_distanceI < len(z):
            FAR_defect = z[MAX_distanceI]
            FAR_point = tuple(cnt[FAR_defect][0])
            return FAR_point
        else:
            return None
    
#BLUE circles for both the Image and Video
def circles(scene, traverse, paint_scene,draw):
    if traverse is not None:
        for i in range(len(traverse)):
            cv.circle(scene, traverse[i], 4, [255, 0, 0], -1)
        for x in range(len(draw)):
            cv.circle(paint_scene, draw[x], 4, [255, 0, 0], -1)
            

            
#Putting farthest (RED CIRCLE), centroid (GREEN CIRCLE), histogram
def image_maker(scene,data,paint_scene):
    image_mask = histogram_maker(scene,data)
    contour_list = contour(image_mask)
    contour_max = maximum(contour_list)
    
    #Looking for centroid of the hand
    moment = cv.moments(contour_max)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        contour_centroid = cx, cy
    else:
        contour_centroid = None
        
    cv.circle(scene, contour_centroid, 5, [0,255,0], -1)
    
    if contour_max is not None:
        hull = cv.convexHull(contour_max, returnPoints=False)
        defect = cv.convexityDefects(contour_max, hull)
        far = farthest(defect, contour_max, contour_centroid)
        print("Centroid : " + str(contour_centroid) + ", farthest Point : " + str(far))
        cv.circle(scene, far, 5, [0, 0, 255], -1)
        if len(traverse) < 20:
            traverse.append(far)
        else:
            traverse.pop(0)
            traverse.append(far)
        draw.append(far)
        
        circles(scene,traverse,paint_scene,draw)


def main():
    global data
    is_created = False
    
    #video_src[0 or 1] 0 for laptop webcam and 1 for external
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[1]
    except:
        video_src = 1
        
    cam = create_capture(video_src)
        
    while True:
        #Loading of image
        screenrect = ImageGrab.grab()
        screenrect_np = np.array(screenrect)
        corrected_scr_np = cv.cvtColor(screenrect_np,cv.COLOR_BGR2RGB)
        paint_scene = cv.imread('Picture.jpg')
        instruct_scene = cv.imread('Instructions.jpg')
        key = cv.waitKey(1)
        _, scene = cam.read()
        
        
        #PRESS E to scan you hand
        if key & 0xFF == ord('e'):
            is_created = True
            data = hand_data(scene)
            
        if is_created:
            image_maker(scene,data,paint_scene)
            
        else:
            scene = scan_region(scene)
        
        cv.imshow("Finger Detection",scene)
        cv.imshow("Paint", paint_scene)
        cv.imshow("Recording",corrected_scr_np)
        cv.imshow("Instructions",instruct_scene)
        screencap.write(corrected_scr_np)
    
        #PRESS Z to clear all
        if key & 0xFF == ord('z'):
            while len(draw) != 0:
                draw.pop()
        
        #PRESS 'esc' to kill the program
        if key == 27:
            break
        
    print('Exited')

    
if __name__ == '__main__':
    main()
    screencap.release()
    cv.destroyAllWindows()
    