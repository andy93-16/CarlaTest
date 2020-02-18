#!/usr/bin/env python

import cv2
import logging
import math
import motion_planning

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

#Define method to process frame in road centering mode
def road_centering(frame):
    edges_road = detect_edges(frame,np.array([244, 35, 232]))     #sidewalk colour
    cropped_edges = region_of_interest(edges_road,0) #mask setting
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = close_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    a, b=display_heading_line(lane_lines_image,steering_angle_detection(frame,lane_lines))
    return motion_planning.stabilize_steering_angle_P(a,len(lane_lines)),b

#Define method to process frame in road centering mode
def lane_centering(frame,prev_steering_angle):
    edges_lane = detect_edges(frame, np.array([128, 64, 128]))   #road colour
    cropped_edges = region_of_interest(edges_lane,1) #mask setting
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = close_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    a, b = display_heading_line(lane_lines_image, steering_angle_detection(frame, lane_lines))
    #different kp for lane_centering respect base k
    return motion_planning.stabilize_steering_angle_P(a,len(lane_lines),prev_steering_angle,kp_two=0.06,kp_one=0.29),lane_lines_image


#Operating Edge Detection
def detect_edges(frame,city_palette):
    # filter for palette
    mask = cv2.inRange(frame,city_palette,city_palette) #range specify colour in mask to find
    # detect edges
    edges = cv2.Canny(mask, 200, 400) #200,400 #canny apply mask and threshold
    return edges

#Operating filter with mask to avoid noise
def region_of_interest(edges,mask_index):
    height, width = edges.shape

    mask_configs =[[(0, height*3/5),(width/2,height/2),
        (width, height*3/5),(width, height),(0, height)],
        [ (0, height*5/7),(width / 2, height * 1 / 3),
        (width, height*5/7),(0, height)]]

    mask = np.zeros_like(edges)
    polygon=np.array([mask_configs[mask_index]],np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

#Operating line segments detection with Hough Lines P
def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)
    return line_segments
def close_slope_intercept(frame, line_segments):
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((-slope,-intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((-slope,-intercept))
    if len(left_fit) > 0:
        left_intercepts=[]
        for elem in left_fit:
            left_intercepts.append(elem[1])
        i = left_intercepts.index(min(left_intercepts))
        lane_lines.append(make_points(frame,(-left_fit[i][0],-left_fit[i][1])))
    if len(right_fit) > 0:
        right_intercepts = []
        for elem in right_fit:
            right_intercepts.append(elem[1])
        i = right_intercepts.index(max(right_intercepts))
        lane_lines.append(make_points(frame,(-right_fit[i][0],-right_fit[i][1])))
    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines

#function to make point display on a certain window
def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line

    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

#create image with line_segment
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return line_image

#function to process the angle between line and Y axis
def steering_angle_detection(frame,lane_lines):
    height, width, _ = frame.shape
    if len(lane_lines) == 2 :
      left_x1,left_y1, left_x2,left_y2 = lane_lines[0][0]
      right_x1,right_y1, right_x2,right_y2 = lane_lines[1][0]
      x1 = (left_x1 + right_x1)/2
      y1 = (left_y1 + right_y1)/2
      x2 = (left_x2 + right_x2)/2
      y2 = (left_y2 + right_y2)/2
      if x1-x2 != 0 :
        angle_to_mid_radian = math.tan((x2 - x1)/(y2 - y1)) #need to be atan((y2-y1)/(x2-x1))
      else :
        angle_to_mid_radian = 0.0

    elif len(lane_lines) == 1:
      x1,y1,x2,y2 = lane_lines[0][0]
      angle_to_mid_radian = -math.atan((x2 - x1) / (y2 - y1))  # angle (in radian) to center vertical line

    #elif len(lane_lines) == 0:


    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    return angle_to_mid_deg/90

#create image with line display angle detected
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry
    x1 = int(width / 2)
    y1 = int(height)
    y2 = int(y1 / 2)
    x2 = int(x1 + y2*math.sin(steering_angle))
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return steering_angle,heading_image


