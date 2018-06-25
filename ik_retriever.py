#! /usr/bin/env python

import rospy
import logging
import argparse
from trac_ik_python.trac_ik import IK

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='odom_receiver')
parser.add_argument('--maxIter', '-mi', type=int, default='50',
						help='max num iterations' )
parser.add_argument('--plot_state', '-ps', action='store_true', default=False,
						help='plot nominal trajectories' )
parser.add_argument('--save_fig', '-sf', action='store_false', default=True,
						help='save plotted figures' )
parser.add_argument('--silent', '-si', action='store_true', default=True,
						help='max num iterations' )
args = parser.parse_args()


urdf = rospy.get_param('/robot_description')
# print("urdf: ", urdf)
ik_solver = IK("first_joint", "seventh_joint", urdf=urdf)

seed_state = [0.0]*ik_solver.number_of_joints

ik_sol = ik_solver.get_ik(sed_state,
				0.45, 0.1, 0.3,  # X, Y, Z
				0.0, 0.0, 0.0, 1.0)  # QX, QY, QZ, QW
print 'ik_sol ', ik_sol

if __name__ == '__main__':
	if args.silent:
		log_level = rospy.INFO
	else:
		log_level = rospy.DEBUG
	try:
		rospy.init_node('trajectory_optimization',
						disable_signals=False, anonymous=True,
						log_level=log_level)
	except KeyboardInterrupt:
		LOGGER.critical("shutting down ros")
