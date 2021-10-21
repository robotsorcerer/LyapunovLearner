import struct
from tampy_common import MAX_JOINT_NUM

# Recieve from Torobo

RX_JOINT_PACKSTR = '4BIBH13f'
RX_PACKET_PACKSTR = '<BBQQf' + RX_JOINT_PACKSTR * MAX_JOINT_NUM + 'H'
RX_SIZE = struct.calcsize(RX_PACKET_PACKSTR)


class RX_JOINT(object):
    def __init__(self, b):
        (
            self.type,
            self.comStatus,
            self.systemMode,
            self.ctrlMode,
            self.ewStatus,
            self.trjStatus,
            self.trjViaRemain,
            self.refCurrent,
            self.refPosition,
            self.refVelocity,
            self.refTorque,
            self.current,
            self.position,
            self.velocity,
            self.torque,
            self.temperature,
            self.kp,
            self.ki,
            self.kd,
            self.damper,
        ) = b


class RX(object):
    def __init__(self, b):
        (
            self.header1,
            self.header2,
            self.timestamp,
            self.host_timestamp,
            self.duration,
        ) = b[:5]
        self.joints = [RX_JOINT(b[5+i*20:25+i*20]) for i in range(MAX_JOINT_NUM)]
        self.crc = b[-1]
