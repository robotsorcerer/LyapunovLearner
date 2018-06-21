import os
import sys
import time
import rospy
import logging
import numpy as np
from os.path import join, expanduser
from trac_ik_python.trac_ik_wrap import TRAC_IK

sys.path.append(join(expanduser('~'), 'catkin_ws', 'src', 'torobo', 'tampy') )
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

		return positions, velocities

	def execute(self, x0, XT, stab_handle, opt_exec):
		"""function runs motions learned with SEDs
			xd = f(x)

			where x is an arbitrary d dimensional variable and xd is the first time derivative
		"""
		xvel_des = np.zeros_like(XT)
		ik_solver = TRAC_IK("link0", "link7", self.urdf,
							0.005,  # default seconds
							1e-5,   # default epsilon
							"Speed")

		qinit = [0.] * 7  #Initial status of the joints as seed.
		x = y = z = 0.0
		rx = ry = rz = 0.0
		rw = 1.0
		bx = by = bz = 0.001
		brx = bry = brz = 0.1

		while True:
			xpos_cur, xvel_cur = self.get_state()
			print(np.array(xpos_cur).shape, np.array(xvel_cur).shape)
			xpos_next = xpos_cur + xvel_des *  opt_exec['dt']

			xvel_des, u = stab_handle(xpos_next - XT)
			xvel_des   = xvel_des + u

			rospy.loginfo(' constructing ik solution ')
			print(' xpos_next: ', xpos_next.shape)
			print(xpos_next)
			num_solutions_found = 0
			for i in range(xpos_next.shape[-1]):
				x, y, xd, yd = xpos_next[:, i]
				ini_t = time.time()
				sol = ik_solver.CartToJnt(qinit,
										  x, y, z,
										  rx, ry, rz, rw,
										  bx, by, bz,
										  brx, bry, brz)
				fin_t = time.time()
				call_time = fin_t - ini_t

				if sol:
					print("X, Y, Z: " + str( (x, y, z) ))
					print("SOL: " + str(sol))
					print
					num_solutions_found += 1

			print
			print("Found " + str(num_solutions_found) + " solutions")
			print

			self.set_position(qinit)

			if np.linalg.norm((xpos_next - xpos_cur), ord=2) > opt_exec['stop_tol']:
				logger.debug('robot reached tolerance; schtoppen')
				break




		return xpos_next, xvel_des


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
