#!/usr/bin/env python
from __future__ import print_function
import sys
import time
import math
import rospy, rospkg
import logging

rospack = rospkg.RosPack()
lyap = rospack.get_path('lyapunovlearner')

sys.path.append('/home/olalekan/catkin_ws/src/torobo/tampy')
from tampy.tampy import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


"""
This script basically moves the bot to the home position, then moves the robot to a near nonlinear pose
The joint angles and joint angle velocities are collected in the process.
We then parameterize the dataset by a control lyapunov function via convex programming
The rest of the task consists of imitation learning the task via DP etc
"""
# In degrees not rads.
def moveto_zero():
    # note that these are position in joint angles
    tampy = Tampy()
    tampy.move_to([0.0] * 7)

def moveto_goalstate():
    tampy = Tampy()
    goal_positions = [0.0, -45.0, 45.0, 60.0, 35.0, 40.0, 10.0  ]
    tampy.move_to(goal_positions)

if __name__ == '__main__':
    logger.debug("moving arm to home position")
    moveto_zero()
    time.sleep(5)
    logger.debug("moving to goal position")
    moveto_goalstate()
