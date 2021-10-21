import numpy as np
import cv2
import atexit
import serial
import struct
import time
import os, sys
from tampy_common import *
from multiprocessing import Process, Array, Value
from rx import RX, RX_SIZE, RX_PACKET_PACKSTR
from tx import TX
from torobo_forward import forward_kinematics
from environments.camera_environment import Camera

def read_loop(ser, packet, init_flag, exit_flag):
	latest_time = time.time()
	freq = 100.0
	while exit_flag.value == 0:
		# rx
		try:
			buff = ser.read(3600)
		except serial.SerialException:
			continue
		except:
			with init_flag.get_lock():
				init_flag.value = -1
			raise

		header_position = [i for i, (b1, b2) in enumerate(zip(buff, buff[1:]))
						   if ord(b1) == HEADER1 and ord(b2) == HEADER2][-2]
		# TODO: exception if header is not found

		packet_buff = buff[header_position:header_position + RX_SIZE]
		# check crc
		crc = struct.unpack(CRC_PACKSTR, packet_buff[-CRC_SIZE:])[0]
		if crc_fun(packet_buff[:-CRC_SIZE]) == crc:
			packet[:] = packet_buff

			with init_flag.get_lock():
				init_flag.value = 1
		else:
			print('crc discrepancy')
		time.sleep(max(0, latest_time + 1.0 / freq - time.time()))


packet = Array('c', RX_SIZE)
init_flag = Value('i', 0)
exit_flag = Value('i', 0)
established_flag = False
ser = None


def terminate_process():
	exit_flag.value = 1


atexit.register(terminate_process)


# share serial instance / process among multiple environments
def establish_connection():
	global established_flag, ser
	if established_flag:
		return
	try:
		ser = serial.Serial(
			'/dev/torobo',
			3000000,
			xonxoff=0,
			bytesize=serial.EIGHTBITS,
			timeout=None
		)
	except:
		raise

	established_flag = True
	Process(
		target=read_loop,
		args=(ser, packet, init_flag, exit_flag)
	).start()


class ToroboException(Exception):
	def __init__(self, tampy):
		super(ToroboException, self).__init__()
		self.tampy = tampy


class Tampy(object):
	def __init__(self, reset_error=True):
		if not established_flag:
			establish_connection()
		assert self.wait_for_first_rx()
		if reset_error:
			self.send(ORDER_RESET)
		if not self.check_error():
			raise ToroboException(self)

		#file opne
		dir_name_tmp = os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]
		dir_name = '/'.join(dir_name_tmp)
		file_path = dir_name + '/data/'
		if not os.path.exists(file_path):
			os.makedirs(file_path)
		self.save_joint = file_path + 'state_joint.npy'
		# self.save_camera = open(dir_name + '/data/' + 'state_camera.npy', 'wb')


	def start_timer(self):
		self.start_time = time.time()


	def wait_for_first_rx(self):
		interval = COMM_TIME
		timeout = 10
		while timeout > 0:
			timeout -= 1
			time.sleep(interval)
			with init_flag.get_lock():
				if init_flag.value == 1:
					return True
		return False


	def check_error(self):
		rx = self.get_latest_rx()
		error_status = [
			(j_id, j.ewStatus) for j_id, j in enumerate(rx.joints) if j.ewStatus != 0
		]
		if len(error_status) == 0:
			return True
		else:
			print('detected error in joint(s):')
			for (joint_id, es) in error_status:
				print(
					'J' +
					format(joint_id+1, '#d') + ' ' +
					format(es >> 16, '#016b') + ' ' +
					format(es & 0xFFFF, '#016b')
				)
			return False


	def get_latest_rx(self):
		assert init_flag > 0
		return RX(
			struct.unpack(RX_PACKET_PACKSTR, bytearray(packet))
		)


	def send_currents(self, currents, joint_id=None):
		tx = TX()
		if joint_id == None:
			for joint_id in range(MAX_JOINT_NUM):
				tx.joints[joint_id].joint_order = ORDER_CURRENT
				tx.joints[joint_id].value1 = currents[joint_id]
		else:
			tx.joints[joint_id].joint_order = ORDER_CURRENT
			tx.joints[joint_id].value1 = currents[0]
		self.send_with_confirmation(tx)


	def send_points(self, positions, duration, joint_id=None):
		tx = TX()
		if joint_id == None:
			for joint_id in range(MAX_JOINT_NUM):
				tx.joints[joint_id].joint_order = ORDER_TRAJ_VIA_APPEND_PT_SIN
				tx.joints[joint_id].value1 = positions[joint_id]
				tx.joints[joint_id].value2 = duration
		else:
			tx.joints[joint_id].joint_order = ORDER_TRAJ_VIA_APPEND_PT_SIN
			tx.joints[joint_id].value1 = positions[0]
			tx.joints[joint_id].value2 = duration
		self.send_with_confirmation(tx)


	def send(self, order, joint_id=None, value1=0.0, value2=0.0):
		tx = TX()
		if joint_id == None:
			for joint_id in range(MAX_JOINT_NUM):
				tx.joints[joint_id].joint_order = order
				tx.joints[joint_id].value1 = value1
				tx.joints[joint_id].value2 = value2
		else:
			tx.joints[joint_id].joint_order = order
			tx.joints[joint_id].value1 = value1
			tx.joints[joint_id].value2 = value2
		self.send_with_confirmation(tx)


	def send_with_confirmation(self, tx):
		timestamp = tx.timestamp
		retry_cnt = 3
		while retry_cnt > 0:
			ser.write(tx.to_bytes())
			if self.wait_for_confirmation(timestamp):
				break
			retry_cnt -= 1
		assert retry_cnt > 0


	def wait_for_confirmation(self, timestamp):
		write_timeout = 7
		while write_timeout > 0:
			rx = self.get_latest_rx()
			if rx.host_timestamp == timestamp:
				return True
			#maximum speed for HostController to MasterController: 20msec
			time.sleep(COMM_TIME)
			write_timeout -= 1
		print('master controller did not confirm the last command')
		return False


	def wait_for_accomplish(self, motion_time, duration, disp=False):
		rx = self.get_latest_rx()
		rx_timestamp = rx.timestamp
		previous_time = time.time() - self.start_time
		self.current_time = time.time() - self.start_time
		log_count_current = 1

		currents_tmp_buf = np.zeros(len([j.current for j in rx.joints]))	#currents
		currents_tmp_buf[:] = self.state[15:22].astype(np.float32)	#initial value of currents

		traj_timeout = True
		while traj_timeout > 0:
			self.current_time = time.time() - self.start_time

			rx = self.get_latest_rx()
			if(rx_timestamp!=rx.timestamp):
				rx_timestamp = rx.timestamp
				log_count_current = log_count_current + 1
				self.get_state_current()
				currents_tmp_buf[:] = currents_tmp_buf[:] + self.state_currents[:]
				if disp:
					print(['{0:3.2f}'.format(j.position) for j in rx.joints])

			if(self.current_time - previous_time > self.log_time):
				self.log_count = self.log_count +1

				previous_time = time.time() - self.start_time
				self.get_state()
				# self.get_state_with_camera()
				self.state_buf[self.log_count,:] = self.state[:].astype(np.float32)
				# self.frame_buf[self.log_count,:,:,:] = self.frame[:,:,:].astype(np.float32)
				self.state_buf[self.log_count,15:22] = (currents_tmp_buf[:]/log_count_current).astype(np.float32)

				currents_tmp_buf = np.zeros(len([j.current for j in rx.joints]))	#currents
				log_count_current = 0

			if(time.time()- self.start_time > motion_time):
				traj_timeout = False
				rx = self.get_latest_rx()
				print('{0:3.4f}'.format(time.time()-self.start_time))
				print(['{0:3.2f}'.format(j.position) for j in rx.joints])

				self.logger()
				# if(time.time()- self.start_time > duration):
				# 	print('logging')
				# 	self.logger()


	def log_buf(self, duration):
		rx = self.get_latest_rx()
		self.log_time = 0.1
		self.log_count = 0
		self.current_time = time.time() - time.time()

		# self.get_state_with_camera()
		self.get_state()
		self.state_buf  = np.zeros(int(duration/self.log_time+1) * (self.state.shape[0])).reshape(int(duration/self.log_time+1), (self.state.shape[0]))
		# self.frame_buf  = np.zeros(int(duration/self.log_time+1) * (self.frame.shape[0]) * (self.frame.shape[1]) * (self.frame.shape[2])).reshape(int(duration/self.log_time+1), (self.frame.shape[0]), (self.frame.shape[1]), (self.frame.shape[2]))

		self.get_state()
		self.state_buf[self.log_count,:] = self.state[:].astype(np.float32)	#initial value
		# self.frame_buf[self.log_count,:,:,:] = self.frame[:,:,:].astype(np.float32)	#initial value


	def start_joints(self, joint_id=None):
		self.send(ORDER_RUN_MODE, joint_id=joint_id, value1=CTRL_MODE_TRAJDYNAMICS)
		#for i in range(MAX_JOINT_NUM):
		#	self.send(ORDER_PARAM_KI, joint_id=i, value1=0.0)
		# according to Torobo, default gain is not enough.
		#self.send(ORDER_PARAM_KP, joint_id=2, value1=50.0)
		#self.send(ORDER_PARAM_KP, joint_id=6, value1=50.0)
		self.send(ORDER_TRAJ_VIA_CLEAR, joint_id=joint_id)


	def move_joints(self, positions, joint_id=None, each_time=5.0, motion_time=5.0, duration=5.0):
		#self.send(ORDER_TRAJ_VIA_CLEAR, joint_id=joint_id)
		self.send_points(positions, each_time, joint_id=joint_id)
		self.send(ORDER_TRAJ_CTRL_START, joint_id=joint_id)
		self.wait_for_accomplish(motion_time, duration)


	def start_currents(self, joint_id=None):
		self.send(ORDER_RUN_MODE, joint_id=joint_id, value1=CTRL_MODE_CURRENT)
		self.send(ORDER_TRAJ_VIA_CLEAR, joint_id=joint_id)


	def move_currents(self, currents, joint_id=None, motion_time=5.0, duration=5.0):
		self.send_currents(currents, joint_id=joint_id)
		self.send(ORDER_TRAJ_CTRL_START, joint_id=joint_id)
		self.wait_for_accomplish(motion_time, duration)


	def servo_on(self, joint_id=None):
		print('servo on')
		self.send(ORDER_SERVO_ON, joint_id=joint_id)


	def servo_off(self, joint_id=None):
		print('servo off')
		self.send(ORDER_SERVO_OFF, joint_id=joint_id)


	def get_state(self):
		rx = self.get_latest_rx()
		positions = [j.position for j in rx.joints]
		velocities = [j.velocity for j in rx.joints]
		currents = [j.current for j in rx.joints]
		T_mat = np.array([forward_kinematics(positions)])
		endeffector = (T_mat[:, 0:3, 3] / 1000.0).reshape(3)
		self.current_time = time.time() - time.time()
		# print('positions ', len(positions), len(velocities), len(currents), endeffector.shape)
		self.state = np.hstack([self.current_time, positions, velocities, currents, endeffector])


	def get_state_current(self):
		rx = self.get_latest_rx()
		self.state_currents = [j.current for j in rx.joints]


	def get_state_with_camera(self):
		# self.frame = self.cam.camera_update(self.cap)
		self.get_state()


	def logger(self):
		print('logging')
		# self.get_state_with_camera()
		self.get_state()

		with open(self.save_joint, 'a') as foo:
			np.save(foo, self.state_buf)

	def camera_launch(self):
		self.cam = Camera()
		self.cap = self.cam.camera_establish()


	def camera_display(self):
		self.p = Process(
			target=self.cam.display
		)
		self.p.start()
