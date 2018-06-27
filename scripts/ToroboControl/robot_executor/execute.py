from __future__ import print_function
import os
import sys
import time
import rospy
import logging
import numpy as np
from config import ik_config
from os.path import join, expanduser
from torobo_forward import forward_kinematics
from trac_ik_python.trac_ik_wrap import TRAC_IK

sys.path.append(join(expanduser('~'), 'Documents', 'LyapunovLearner', 'ToroboTakahashi') )
sys.path.append(join(expanduser('~'), 'Documents', 'LyapunovLearner', 'ToroboTakahashi', 'tampy') )
from tampy.tampy import Tampy
from tampy.tampy import ORDER_SERVO_ON, ORDER_SERVO_OFF, ORDER_RUN_MODE, CTRL_MODE_CURRENT

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ToroboExecutor(object):
	def __init__(self, home_pos, urdf):
		"""
		 in update(self, currents) currents are inindividual currents to each joint
		 According to Ryo-san, the max current is 0.5A and you should start with 0.05A or sth and gradually increase
		"""
		self.tampy = Tampy()
		self.urdf  = urdf
		self.home_pos = home_pos
		self.control_freq = 30.0
		self.latest_control = time.time()

	def set_position(self, positions):
		self.tampy.move_to(positions)

	def update(self, currents):
		self.tampy.send_currents(currents)
		time.sleep(max(
			self.latest_control + 1.0 / self.control_freq - time.time(),
			0
		))
		self.latest_control = time.time()
		return self.get_state()

	def get_state(self):
		rx = self.tampy.get_latest_rx()
		positions = [j.position for j in rx.joints]
		velocities = [j.velocity for j in rx.joints]
		T_mat = np.array([forward_kinematics(positions)])
		endeffector = (T_mat[:, 0:3, 3] / 1000.0).reshape(3)

		# print('T_mat: ', T_mat)
		# print('end_eff: ', endeffector)

		return positions, velocities, endeffector

	def cart_to_joint(self, cart_pos, cart_vel, timeout=0.5, check=True):

		ik_solver = TRAC_IK("link0", "link7", self.urdf,
							ik_config['ik_timeout'],  # default seconds
							1e-5,   # default epsilon
							"Speed")

		if check: # check that ik solver is actually working
			print("Number of joints:", ik_solver.getNrOfJointsInChain())
			print("Joint names: ", ik_solver.getJointNamesInChain(self.urdf))
			print("Link names:", ik_solver.getLinkNamesInChain())

		qinit = [0.] * 7  #[-1.24, -8.16, -4.30, 68.21, -17.01, 59.76, 0.03]#
		x, y, z, xd, yd, zd = cart_pos
		rx = ry = rz = 0.0
		rw = 1.0
		bx = by = bz = 0.001
		brx = bry = brz = 0.1

		avg_time = num_solutions_found = 0
		num_waypts   = 2

		for i in range(ik_config['ik_trials_num']):
			ini_t = time.time()
			pos = ik_solver.CartToJnt(qinit,
									  x, y, z,
									  rx, ry, rz, rw,
									  bx, by, bz,
									  brx, bry, brz)
  			vel = ik_solver.CartToJnt(qinit,
  									  cart_vel[:3],
  									  rx, ry, rz, rw,
  									  bx, by, bz,
  									  brx, bry, brz)
			fin_t = time.time()
			call_time = fin_t - ini_t
			avg_time += call_time

			if pos and vel:
				num_solutions_found += 1

		if check:
			print()
			print("Found " + str(num_solutions_found) + " solutions")
			print("X, Y, Z: " + str( (x, y, z) ))
			print("POS: " , list(pos))
			print("VEL: " , list(vel))
			print("Average IK call time: %.4f mins"%(avg_time*60))
			print()
		return list(pos), list(vel)

	@staticmethod
	def expand(x, axis):
		return np.expand_dims(x, axis)

	def execute(self, data, stab_handle, opt_exec):
		"""
		    function runs motions learned with SEDs
			xd = f(x)

			where x is an arbitrary d dimensional variable and xd is the first time derivative
		"""
		x0       = data[:, 0]
		x_des       = data[:, 356]

		d = data.shape[0]/2

		while not rospy.is_shutdown():
			t1 = time.time()
			# these are in joint coordinates
			q_cur, qdot_cur, x_cur = self.get_state()

			t2 = time.time()
			xdot_cur = x_cur / (t2-t1)

			# compute f
			f, u = stab_handle(data - np.expand_dims(x_des, 1))
			xvel_des = f + u

			print('xvel_des: ', xvel_des.shape)
			# get next state
			x_next = self.expand(x_cur[:d], 1) + xvel_des *  opt_exec['dt']#(t2-t1) #o
			rospy.logdebug(' constructing ik solution for robot trajectory')

			# assemble state for joint trajectory
			print('x_next: ', x_next)

			diff = np.linalg.norm((x_next - self.expand(x_cur[:d], 1)), ord=2)
			rospy.logdebug('diff: '.format(diff))

			# self.set_position(qinit)

			if diff > opt_exec['stop_tol']:
				rospy.logdebug('robot reached tolerance; schtoppen')
				break




		return x_next, xvel_des


		def __enter__(self):
			self.set_position(self.home_pos)
			self.tampy.send(ORDER_RUN_MODE, value1=CTRL_MODE_CURRENT)
			self.tampy.send(ORDER_SERVO_ON)
			self.latest_control = time.time()
			return self

		def __exit__(self, type, value, tb):
			self.tampy.send_currents([0] * 7)
			# TODO: why do we need multiple calls to kill?
			for _ in range(3):
				self.tampy.send(ORDER_SERVO_OFF)
