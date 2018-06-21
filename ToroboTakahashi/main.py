#!/usr/bin/env python

import os
import cv2
import time
import argparse
import numpy as np
from tampy.tampy import Tampy
from os.path import expanduser, join
from torobo_forward import forward_kinematics


parser = argparse.ArgumentParser(description='teacher')
parser.add_argument('--teach', '-te', action='store_true', default=True)
parser.add_argument('--turn_off_currents', '-co', action='store_false', default=False)
args = parser.parse_args()


tampy = Tampy()

def move_home(duration):
	# tampy = Tampy()
	home = [0.00]*7

	tampy.servo_on()
	each_time = 5.00
	motion_time = 5.00
	joint_angle = 0.0

	tampy.start_joints()
	tampy.log_buf(duration)
	tampy.start_timer()
	tampy.move_joints(home, each_time=each_time, motion_time=motion_time, duration=duration)

def move_teach(duration):
	# tampy = Tampy()

	Red    = [14.5, 47.8, -6.6, 73.3, 6.3, 50.9, -0.3]
	Pipe = [30.23, -2.28, -9.68, 48.09, -23.25, 68.50, -0.03]
	Trumpet = [30.15, -8.90, 0.12, 59.71, 4.80, 83.20, -0.01]
	Pos3 = [0.0, 47.8, -6.6, 73.3, 6.3, 50.9, -0.3]
	Pos4 = [10.0, 47.8, -6.6, 73.3, 6.3, 50.9, -0.3]
	Pos5 = [20.0, 47.8, -6.6, 73.3, 6.3, 50.9, -0.3]

	tampy.servo_on()
	each_time = 5.00
	motion_time = 5.00
	joint_angle = 0.0

	tampy.start_joints()
	tampy.log_buf(duration)
	tampy.start_timer()
	#tampy.move_joints(Yellow, each_time=each_time, motion_time=motion_time, duration=duration)
	tampy.move_joints(Pipe, each_time=each_time, motion_time=motion_time, duration=duration)
	move_home(duration)
	time.sleep(1)
	tampy.move_joints(Trumpet, each_time=each_time, motion_time=motion_time, duration=duration)

def move_gene():
	#self.current_time, positions, velocities, currents, endeffector
	#joint_state = np.load("/home/takahashi/PFN_repo/torobo_basic/data/Yellow/state_joint.npy")
	#joint_state = np.load("/home/takahashi/PFN_repo/torobo_basic/data/Red/state_joint.npy")
	# joint_state = np.load("/home/takahashi/PFN_repo/torobo_basic/data/Pos5/state_joint.npy")

	tampy.camera_launch()
	tampy.servo_on()

	step_time = 0.1
	duration = 5.0
	tampy.start_currents()
	tampy.log_buf(duration)
	tampy.start_time()
	for i in range(1, 50):
		tampy.move_currents(joint_state[i,15:22], motion_time=step_time*(i+1), duration=duration)
	tampy.servo_off()

def move_joints_test():
	tampy.camera_launch()
	tampy.servo_on()

	duration = 9.00

	tampy.log_buf(duration)
	tampy.start_time()

	each_time = 3.00
	motion_time = 3.00
	joint_angle = 30.0
	tampy.start_joints()
	tampy.move_joints([joint_angle]*7, each_time=each_time, motion_time=motion_time, duration=duration)

	each_time = 1.50
	motion_time = 4.50
	joint_angle = 15.0
	tampy.start_joints()
	tampy.move_joints([joint_angle]*7, each_time=each_time, motion_time=motion_time, duration=duration)

	each_time = 1.50
	motion_time = 6.00
	joint_angle = 0.0
	tampy.start_joints()
	tampy.move_joints([joint_angle]*7, each_time=each_time, motion_time=motion_time, duration=duration)
	#tampy.start_currents()
	#tampy.move_currents([joint_angle]*7, duration=duration, motion_time=duration)
	#tampy.servo_off()

def turn_off_currents():
	# tampy.camera_launch()
	tampy.servo_on()
	tampy.start_currents()
	tampy.start_timer()
	duration = 100.00

	tampy.log_buf(duration)

	motion_time = 700.00
	joint_angle = 0.0

	for id in range(7):
		tampy.start_currents(joint_id=id)
		tampy.move_currents([0.0]*7, joint_id=id, duration=duration, motion_time=duration)
	tampy.servo_off()


def move_reset():
	tampy.camera_launch()
	tampy.servo_on()

	duration = 5.00500

	tampy.log_buf(duration)
	tampy.start_time()

	each_time = 5.00
	motion_time = 5.00
	joint_angle = 0.0
	tampy.start_joints()
	tampy.move_joints([joint_angle]*7, each_time=each_time, motion_time=motion_time, duration=duration)

if __name__ == '__main__':
	filepath = join(expanduser('~'), 'Documents', 'LyapunovLearner', 'ToroboTakahashi', 'data')
	name = 'state_joint.npy'
	filename = join(filepath, name)

	if os.path.isfile(filename):
		os.remove(filename)

	tampy.get_state()
	print('state: ', abs(sum(tampy.state[1:8])))
	time.sleep(4)

	duration = 1000.00

	if abs(sum(tampy.state[1:8])) > 0.5 : # already in home state
		move_home(duration)
		duration += 1000

	if args.turn_off_currents:
		turn_off_currents()
	elif args.teach:
		move_teach(duration)
		time.sleep(2)

	move_home(duration)

	# move_gene()
	#move_reset()
