#! /usr/bin/env python
from __future__ import print_function
import os
import sys
import time
import logging
sys.path.append('/home/olalekan/catkin_ws/src/torobo/tampy')
from tampy.tampy import Tampy
from tampy.tampy import ORDER_SERVO_ON, ORDER_SERVO_OFF, ORDER_RUN_MODE, CTRL_MODE_CURRENT


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ToroboEnvironment(object):
    def __init__(self, home_pos):
        """
         in update(self, currents) currents are inindividual currents to each joint
         According to Ryo-san, the max current is 0.5A and you should start with 0.05A or sth and gradually increase
        """
        self.tampy = Tampy()
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

def do_motions(data_file):
    home_pos = [0.0] * 7
    torobo = ToroboEnvironment(home_pos)

    logger.debug("moving arm to home position")
    torobo.set_position(torobo.home_pos)
    time.sleep(5)

    # logger.debug("shutting off currents to each joint")
    # currents = [0.0]*7
    # torobo.update(currents)

    logger.debug("fetching positions and velocities 1")
    pos, vel = torobo.get_state()
    with open(data_file, 'a') as f:
        f.write("%s\n"%(pos + vel))

    temp_positions = [0.0, -45.0, 45.0, 60.0, 35.0, 40.0, 10.0  ]
    torobo.set_position(temp_positions)
    logger.debug("fetching positions and velocities 2")
    pos2, vel2 = torobo.get_state()
    with open(data_file, 'a') as f:
        f.write("%s\n"%(pos2 + vel2))

    final_position = [0.0, 15.0, 25.0, 30.0, 45.0, 15.0, 20.0 ]
    torobo.set_position(final_position)
    logger.debug("fetching positions and velocities 3")
    pos3, vel3 = torobo.get_state()
    with open(data_file, 'a') as f:
        f.write("%s\n\n"%(pos3 + vel3))


if __name__ == '__main__':
    data_file = "moplan_data.csv"

    if os.path.isfile(data_file):
        os.remove(data_file)

    for _ in range(7):
        logger.info("going on iteration %d"%(_))
        do_motions(data_file)
