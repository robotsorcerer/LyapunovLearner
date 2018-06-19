import os
import sys
import time
import logging
import numpy as np
from os.path import join, expanduser

sys.path.append(join(os.expanduser('~'), 'catkin_ws', 'src', 'torobo', 'tampy') )
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

    def execute(self, XT, stab_handle, opt_exec):
        """function runs motions learned with SEDs
            xd = f(x)

            where x is an arbitrary d dimensional variable and xd is the first time derivative
        """
        while True:
            xpos_cur, xvel_cur = self.get_state()

            # xd[:,i,:] = np.tile(ds_stab(np.squeeze(x[:,i,:])-XT), [d, 1, nbSPoint])
            xvel_des, u = ds_stab(xpos_cur - XT)
            xvel_des   = xvel_des + u


            ### integration
            x[:,i,:] = x[:,i,:] + xd[:,i,:]*opt_exec['dt']



            ### integration ###
            x[:,i+1,:] = x[:,i,:] + xd[:,i,:] * options['dt']
            t[i+1] = t[i]+options['dt']

            i += 1

            # check convergence
            if i > 3 and (np.all(np.all(np.all(abs(xd[:,-1-3:-1,:])))))




        return x, xd


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
