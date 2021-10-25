import zmq
import json
from time import sleep


"""
sanity checks copied from armctl provided by Torobo.
"""


def isInvalidJointId(id):
    list = ["all", "1", "2", "3", "4", "5", "6", "7", "8"]
    sp = id.split("/")
    for s in sp:
        if s not in list:
            return True
    return False


def isInvalidServoState(state):
    list = ["on", "ON", "off", "OFF"]
    if state in list:
        return False
    else:
        return True


def isFloat(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def isInvalidCommand(command):
    isInvalidCommand = False
    if "joint_id" in command.keys():
        if isInvalidJointId(command["joint_id"]):
            print "Invalid Joint ID"
            isInvalidCommand = True
    if "value" in command.keys():
        if not (command["value"].isdigit() or isFloat(command["value"])):
            print "Invalid Value"
            isInvalidCommand = True
    if "pos" in command.keys():
        if not (command["pos"].isdigit() or isFloat(command["pos"])):
            print "Invalid Position"
            isInvalidCommand = True
    if "time" in command.keys() and command["time"] is not None:
        if not (command["time"].isdigit() or isFloat(command["time"])):
            print "Invalid Time"
            isInvalidCommand = True
    if "servo_state" in command.keys():
        if isInvalidServoState(command["servo_state"]):
            print "Invalid Servo State"
            isInvalidCommand = True
    if command == {}:
        isInvalidCommand = True

    return isInvalidCommand


def parse_joint_id(joint_id):
    if joint_id == -1:
        return 'all'
    else:
        return str(joint_id+1)


def move_to_home(joint_id, home_pos):
    commands = [
        {
            "command": "--mode",
            "mode_id": "20",
            "joint_id": parse_joint_id(joint_id)
        },
        # default gain for J7 is not enough :(
        {
            "command": "--kp",
            "joint_id": "7",
            "value": "80.0"
        },
        {
            "command": "--ki",
            "joint_id": "7",
            "value": "2.00"
        },
        {
            "command": "--servo",
            "servo_state": "on",
            "joint_id": parse_joint_id(joint_id)
        },
        {
            "command": "--tc",
            "joint_id": parse_joint_id(joint_id)
        },
    ]
    if joint_id == -1:
        commands += [
            {
                "command": "--tpts",
                "joint_id": parse_joint_id(joint_id_),
                "pos": str(pos),
                "time": "5"
            } for joint_id_, pos in enumerate(home_pos)
        ]
    else:
        commands += [
            {
                "command": "--tpts",
                "joint_id": parse_joint_id(joint_id),
                "pos": str(home_pos[joint_id]),
                "time": "5"
            }
        ]
    commands += [
        {
            "command": "--ts",
            "joint_id": parse_joint_id(joint_id)
        }
    ]

    print("moving to home...")
    for command in commands:
        send_command(command)
        sleep(0.1)

    TRAJ_STATUS = [1, 2, 3]
    while True:
        rs = json.loads(request_state())
        ts = rs['jointState'][joint_id]['trjStatus']
        if ts == 4:
            print("move to home successfully finished")
            break
        elif ts not in TRAJ_STATUS:
            raise Exception
        else:
            sleep(0.1)

    for command in commands:
        send_command({
            "command": "--brake",
            "brake_state": "on",
            "joint_id": "all"
        })


def initialize(joint_id, home_pos):
    move_to_home(-1, home_pos)
    commands = [
        {
            "command": "--mode",
            "mode_id": "2",
            "joint_id": parse_joint_id(joint_id)
        },
        {
            "command": "--servo",
            "servo_state": "off",
            "joint_id": parse_joint_id(joint_id)
        },
        {
            "command": "--servo",
            "servo_state": "on",
            "joint_id": parse_joint_id(joint_id)
        },
    ]
    for command in commands:
        ret = send_command(command)

    if not check_error(ret):
        raise Exception


def finalize(joint_id):
    commands = [
        {
            "command": "--servo",
            "servo_state": "off",
            "joint_id": parse_joint_id(joint_id)
        },
        {
            "command": "--brake",
            "brake_state": "on",
            "joint_id": parse_joint_id(joint_id)
        },
    ]
    for command in commands:
        send_command(command)


def set_torque(torque, joint_id):
    command = {
        "command": "--tor",
        "value": str(torque),
        "joint_id": parse_joint_id(joint_id)
    }
    return send_command(command)


def set_current(current, joint_id):
    command = {
        "command": "--cur",
        "value": str(current),
        "joint_id": parse_joint_id(joint_id)
    }
    return send_command(command)


def set_position(position, joint_id):
    command = {
        "command": "--pos",
        "value": str(position),
        "joint_id": parse_joint_id(joint_id),
    }
    return send_command(command)


def request_state():
    command = {
        "command": "--state"
    }
    return send_command(command)


def send_command(command):
    assert not isInvalidCommand(command)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect('tcp://localhost:5555')
    sock.send(json.dumps(command))
    curState = sock.recv()
    return curState


def check_error(state):
    js = json.loads(state)['jointState']
    error = [ss['ewStatus'] for ss in js]
    if all([ee == 0 for ee in error]):
        return True

    slave_error = [ee / 65536 for ee in error]
    master_error = [ee % 65536 for ee in error]

    print('slave')
    for joint_id, se in enumerate(slave_error):
        print(str(joint_id+1) + ':' + format(se, '#016b'))
    print('master')
    for joint_id, me in enumerate(master_error):
        print(str(joint_id+1) + ':' + format(me, '#016b'))

    return False
