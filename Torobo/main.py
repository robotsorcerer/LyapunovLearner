# !/usr/bin/env python

import os
import cv2
import time
import argparse
import numpy as np
from tampy.tampy import Tampy
from os.path import expanduser, join
from torobo_forward import forward_kinematics


parser = argparse.ArgumentParser(description='teacher')
parser.add_argument('--teach', '-te', type=bool, default=0)
parser.add_argument('--publish', '-pb', type=bool, default=0)
parser.add_argument('--turn_off_currents', '-co', type=bool, default=0)
args = parser.parse_args()

tampy = Tampy()

def move_home(duration):

	home = [0.00]*7

	tampy.servo_on()
	each_time = 5.00
	motion_time = 50.00
	joint_angle = 0.0

	tampy.start_joints()
	tampy.log_buf(duration)
	tampy.start_timer()
	tampy.move_joints(home, each_time=each_time, motion_time=motion_time, duration=duration)

def move_teach(duration):
	tampy = Tampy()
	# original
	Pipe = [-1.24, -8.16, -4.30, 68.21, -17.01, 59.76, 0.03]
	Trumpet = [-0.46, -19.55, 14.36, 78.28, 2.73, 60.06, 0.01]

	# mod to min singularity of joint 6
	Pipe = [-1.24, -8.16, -4.30, 68.21, -17.01, 39.76, 0.03]
	Trumpet = [-0.46, -19.55, 14.36, 78.28, 2.73, 30.06, 0.01]

	tampy.servo_on()
	each_time = 5.00
	motion_time = 10.00
	joint_angle = 0.0

	tampy.start_joints()
	tampy.log_buf(duration)
	tampy.start_timer()

	tampy.move_joints(Pipe, each_time=each_time, motion_time=motion_time, duration=duration)

	move_reset()

	# tampy.move_joints(Trumpet, each_time=each_time, motion_time=motion_time, duration=duration)

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

def turn_off_currents(duration):
	tampy = Tampy()
	# tampy.camera_launch()
	tampy.servo_on()
	tampy.start_currents()
	tampy.start_timer()

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
	tampy.start_timer()

	each_time = 5.00
	motion_time = 5.00
	joint_angle = 0.0
	tampy.start_joints()
	tampy.move_joints([joint_angle]*7, each_time=each_time, motion_time=motion_time, duration=duration)

if __name__ == '__main__':
	filepath = join(expanduser('~'), 'Documents', 'LyapunovLearner', 'ToroboTakahashi', 'data')
	name = 'state_joint.npy'
	filename = join(filepath, name)

	tampy.get_state()
	print('state: ', abs(sum(tampy.state[1:8])))
	time.sleep(4)

	duration = 1000.00

	if args.turn_off_currents:
		# if os.path.isfile(filename):
		# 	os.remove(filename)
		turn_off_currents(duration)

	elif args.teach:

		if abs(sum(tampy.state[1:8])) > 0.5 : # already in home state
			print("moving to home position")
			move_reset()
		if os.path.isfile(filename):
			os.remove(filename)

		move_teach(duration)
		# time.sleep(2)

		move_home(duration)

	# move_gene()
	#move_reset()
