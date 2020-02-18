#!/usr/bin/env python


# Using angle obtain from image processing to stabilize the steering angle
# in case one line missed, need to apply a strong k than two line
def stabilize_steering_angle_P(curr_steering_angle,len,prev_steering_angle,kp_two=0.1,kp_one=0.4):

    alpha=0.7

    if len==2:
        kp=kp_two
    else :
        kp=kp_one

    curr_steering_angle = kp * curr_steering_angle

    stabilized_steering_angle = alpha*curr_steering_angle + (1-alpha)*prev_steering_angle

    return stabilized_steering_angle

