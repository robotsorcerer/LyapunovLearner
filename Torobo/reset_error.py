from tampy.tampy import Tampy, ToroboException, ORDER_BRAKE_OFF, ORDER_BRAKE_ON, ORDER_SERVO_OFF, ORDER_RESET
import time


def fix_joint_angle_limit(tampy):
	rx = tampy.get_latest_rx()
	error_joints = [
		(j_id, j) for j_id, j in enumerate(rx.joints)
		if ((j.ewStatus >> 16) & 0x1) != 0
	]
	print error_joints
	if len(error_joints) == 0:
		return True
	for joint_id, joint in error_joints:
		print('fix joint %d' % (joint_id + 1))
		tampy.send(ORDER_SERVO_OFF, joint_id=joint_id)
		tampy.send(ORDER_BRAKE_OFF, joint_id=joint_id)
		time.sleep(5)
		tampy.send(ORDER_BRAKE_ON, joint_id=joint_id)
		tampy.send(ORDER_RESET, joint_id=joint_id)
	return False


def reset_error():
	try:
		tampy = Tampy()
	except ToroboException as e:
		while True:
			if fix_joint_angle_limit(e.tampy):
				tampy = e.tampy
				break
	tampy.camera_launch()
	tampy.servo_on()
	tampy.start_joints()
	tampy.log_buf(duration=5.0)
	tampy.start_timer()
	tampy.move_joints([0.0] * 7, each_time=5.0, duration=5.0, motion_time=5.0)
	tampy.servo_off()
	print('done')


if __name__ == '__main__':
	reset_error()
