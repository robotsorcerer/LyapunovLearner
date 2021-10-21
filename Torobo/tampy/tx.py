import struct
from tampy_common import MAX_JOINT_NUM, HEADER1, HEADER2, CRC_PACKSTR, crc_fun
import time

# Send to Torobo

ORDER_NONE = 0x00
TX_JOINT_PACKSTR = 'BB4f'
TX_MSG_PACKSTR = '<BBQB' + TX_JOINT_PACKSTR * MAX_JOINT_NUM


class TX_JOINT(object):
    def __init__(self, id):
        self.ID = id
        self.joint_order = 0
        self.value1 = 0.0
        self.value2 = 0.0
        self.value3 = 0.0
        self.value4 = 0.0

    def to_list(self):
        return [
            self.ID,
            self.joint_order,
            self.value1,
            self.value2,
            self.value3,
            self.value4
        ]


class TX(object):
    def __init__(self):
        self.joints = [TX_JOINT(id) for id in range(MAX_JOINT_NUM)]
        self.order = ORDER_NONE
        self.timestamp = int(time.time() * 1000)

    def to_bytes(self):
        joint_msg = []
        for joint in self.joints:
            joint_msg += joint.to_list()
        msg = struct.pack(
            TX_MSG_PACKSTR,
            HEADER1,
            HEADER2,
            self.timestamp,  # timestamp
            self.order,  # armorder
            *joint_msg
        )
        crc = struct.pack(CRC_PACKSTR, crc_fun(msg))
        return msg + crc
