#! /usr/bin/env python
# from __future__ import print_statement
import rospy
import logging
import argparse
from trac_ik_python.trac_ik_wrap import TRAC_IK
from numpy.random import random
import time


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


def main():
	urdf = rospy.get_param('/robot_description')
	# print("urdf: ", urdf)
	rospy.loginfo('Constructing IK solver...')
	ik_solver = TRAC_IK("link0", "link7", urdf,
						0.005,  # default seconds
						1e-5,  # default epsilon
						"Speed")
	# print("Number of joints:")
	# print(ik_solver.getNrOfJointsInChain())
	# print("Joint names:")
	# print(ik_solver.getJointNamesInChain(urdf))
	# print("Link names:")
	# print(ik_solver.getLinkNamesInChain())

	qinit = [0.] * 7  #Initial status of the joints as seed.
	x = y = z = 0.0
	rx = ry = rz = 0.0
	rw = 1.0
	bx = by = bz = 0.001
	brx = bry = brz = 0.1

	# Generate a set of random coords in the arm workarea approx
	NUM_COORDS = 200
	rand_coords = []
	for _ in range(NUM_COORDS):
		x = random() * 0.5
		y = random() * 0.6 + -0.3
		z = random() * 0.7 + -0.35
		rand_coords.append((x, y, z))

	# Check some random coords with fixed orientation
	avg_time = 0.0
	num_solutions_found = 0
	for x, y, z in rand_coords:
		ini_t = time.time()
		sol = ik_solver.CartToJnt(qinit,
								  x, y, z,
								  rx, ry, rz, rw,
								  bx, by, bz,
								  brx, bry, brz)
		fin_t = time.time()
		call_time = fin_t - ini_t
		# print "IK call took: " + str(call_time)
		avg_time += call_time
		if sol:
			# print "X, Y, Z: " + str( (x, y, z) )
			print "SOL: " + str(sol)
			num_solutions_found += 1
	avg_time = avg_time / NUM_COORDS

	print
	print "Found " + str(num_solutions_found) + " of 200 random coords"
	print "Average IK call time: " + str(avg_time)
	print

# Check if orientation bounds work
	avg_time = 0.0
	num_solutions_found = 0
	brx = bry = brz = 9999.0  # We don't care about orientation
	for x, y, z in rand_coords:
		ini_t = time.time()
		sol = ik_solver.CartToJnt(qinit,
								  x, y, z,
								  rx, ry, rz, rw,
								  bx, by, bz,
								  brx, bry, brz)
		fin_t = time.time()
		call_time = fin_t - ini_t
		# print "IK call took: " + str(call_time)
		avg_time += call_time
		if sol:
			# print "X, Y, Z: " + str( (x, y, z) )
			# print "SOL: " + str(sol)
			num_solutions_found += 1

	avg_time = avg_time / NUM_COORDS
	print
	print "Found " + str(num_solutions_found) + " of 200 random coords ignoring orientation"
	print "Average IK call time: " + str(avg_time)
	print

	# Check if big position and orientation bounds work
	avg_time = 0.0
	num_solutions_found = 0
	bx = by = bz = 9999.0
	brx = bry = brz = 9999.0
	for x, y, z in rand_coords:
		ini_t = time.time()
		sol = ik_solver.CartToJnt(qinit,
								  x, y, z,
								  rx, ry, rz, rw,
								  bx, by, bz,
								  brx, bry, brz)
		fin_t = time.time()
		call_time = fin_t - ini_t
		# print "IK call took: " + str(call_time)
		avg_time += call_time
		if sol:
			# print "X, Y, Z: " + str( (x, y, z) )
			# print "SOL: " + str(sol)
			num_solutions_found += 1

	avg_time = avg_time / NUM_COORDS

	print
	print "Found " + str(num_solutions_found) + " of 200 random coords ignoring everything"
	print "Average IK call time: " + str(avg_time)
	print

if __name__ == '__main__':
	if args.silent:
		log_level = rospy.INFO
	else:
		log_level = rospy.DEBUG
	try:
		rospy.init_node('trajectory_optimization', log_level=log_level)
		rospy.loginfo('Trajectory optimization node started!')


		# rospy.start()
		# now = rospy.Time.now()
		# zero_time = rospy.Time()
		main()
		rospy.spin()
	except KeyboardInterrupt:
		LOGGER.critical("shutting down ros")
