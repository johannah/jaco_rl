import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dm_control import manipulation
from copy import deepcopy
import numpy as np
import torch
import argparse
import os
import sys
import pickle
import utils
from replay_buffer import ReplayBuffer, compress_frame
import plotting 
import time
from glob import glob

from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from dm_control import viewer
from IPython import embed
from jaco_physics import JacoPhysics
from jaco_safety import JacoSafety,  trim_and_check_pose_safety
random_state = np.random.RandomState(0)

def get_next_frame(frame_env, height, width):
    next_frame = frame_env.physics.render(height=height,width=width,camera_id=args.camera_view)
    return next_frame

def mirror():
    robot_env = JacoPhysics('j2s7s300', robot_server_ip='127.0.0.1', robot_server_port=9030, fence=safety_fence, control_type='position')

    #safe_position, hit = trim_and_check_pose_safety(target_position, safety_fence)
    
    robot_state = robot_env.reset()
    # abs pos in radians
    robot_frame = get_next_frame(robot_env, robot_frame_height, robot_frame_width)
    for i in range(10):
        pos = robot_state[:13]
        action = deepcopy(pos)
        # turn 1st joint
        action[0] += .1
        # TODO - check safety proposed action
        #if jaco_safety_env.task.count_safety_violations(proposed_robot_action):
        robot_next_state = robot_env.step(action)
        robot_next_frame = get_next_frame(robot_env, robot_frame_height, robot_frame_width)
        robot_state = robot_next_state
        time.sleep(1)
if __name__ == "__main__":
    robot_frame_height = 640 
    robot_frame_width = 480 
    safety_fence = {'x':(-.45,.45), 'y':(-1.0, .4), 'z':(.15, 1.2)}
    robot_cam_dim = [robot_frame_height, robot_frame_width, 3]
    mirror()





