#!/usr/bin/env python
## -*- coding: utf-8 -*-
import zmq
from docopt import docopt
import json

__doc__ = """{f}
Usage:
    {f} --state [--joint=<id>]
    {f} --servo <servo_state> [--joint=<id>]
    {f} --brake <brake_state> [--joint=<id>]
    {f} --quit
    {f} --reset [--joint=<id>]
    {f} --mode <mode_id> [--joint=<id>]
    {f} [--kp|--ki|--kd|--da|--cur|--pos|--vel|--tor] <value> [--joint=<id>]
    {f} --tc [--joint=<id>]
    {f} --ts [--joint=<id>]
    {f} --tptl <pos> <time> [--joint=<id>]
    {f} --tpts <pos> <time> [--joint=<id>]
    {f} --tpvt <csv_file>


Options:
    -h --help   show this help message and exit
    --joint=<id>     Select Joint ID
    --state     Show Current Status
    --servo     Servo on/off
    --brake     Brake on/off
    --quit      All servo off
    --reset     Reset joint status
    --mode      Set another control mode
    --kp        Change Kp Value
    --ki        Change Ki Value
    --kd        Change Kd Value
    --da        Change Damper Value
    --cur       Set Reference Current
    --pos       Set Reference Position
    --vel       Set Reference Velocity
    --tor       Set Reference Torque
    --tc        Clear Trajectory Points
    --ts        Start Trajectory Control
    --tptl      Register Linear Trajectory
    --tpts      Register Sin Trajectory
    --tpvt      Register PVT Points Trajectory
""".format(f=__file__)

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
    if command.has_key("joint_id"):
        if isInvalidJointId(command["joint_id"]):
            print "Invalid Joint ID"
            isInvalidCommand = True
    if command.has_key("value"):
        if not (command["value"].isdigit() or isFloat(command["value"])):
            print "Invalid Value"
            isInvalidCommand = True
    if command.has_key("pos"):
        if not (command["pos"].isdigit() or isFloat(command["pos"])):
            print "Invalid Position"
            isInvalidCommand = True
    if command.has_key("time") and command["time"]!=None:
        if not (command["time"].isdigit() or isFloat(command["time"])):
            print "Invalid Time"
            isInvalidCommand = True
    if command.has_key("servo_state"):
        if isInvalidServoState(command["servo_state"]):
            print "Invalid Servo State"
            isInvalidCommand = True
    if command == {}:
        isInvalidCommand = True

    return isInvalidCommand

if __name__=='__main__':
    command = {}

    # Parse Commnd and create command json
    args = docopt(__doc__)
    if args["--state"]:
        command = {
            "command": "--state"
        }
    elif args["--servo"]:
        command = {
            "command": "--servo",
            "servo_state": args["<servo_state>"],
            "joint_id": args["--joint"]
        }
    elif args["--brake"]:
        command = {
            "command": "--brake",
            "brake_state": args["<brake_state>"],
            "joint_id": args["--joint"]
        }
    elif args["--quit"]:
        command = {
            "command": "--quit"
        }
    elif args["--reset"]:
        command = {
            "command": "--reset",
            "joint_id": args["--joint"]
        }
    elif args["--mode"]:
        command = {
            "command": "--mode",
            "mode_id": args["<mode_id>"],
            "joint_id": args["--joint"]
        }
    elif args["--kp"]:
        command = {
            "command": "--kp",
            "value": args["<value>"],
            "joint_id": args["--joint"]
        }
    elif args["--ki"]:
        command = {
            "command": "--ki",
            "value": args["<value>"],
            "joint_id": args["--joint"]
        }
    elif args["--kd"]:
        command = {
            "command": "--kd",
            "value": args["<value>"],
            "joint_id": args["--joint"]
        }
    elif args["--da"]:
        command = {
            "command": "--da",
            "value": args["<value>"],
            "joint_id": args["--joint"]
        }
    elif args["--cur"]:
        command = {
            "command": "--cur",
            "value": args["<value>"],
            "joint_id": args["--joint"]
        }
    elif args["--pos"]:
        command = {
            "command": "--pos",
            "value": args["<value>"],
            "joint_id": args["--joint"]
        }
    elif args["--vel"]:
        command = {
            "command": "--vel",
            "value": args["<value>"],
            "joint_id": args["--joint"]
        }
    elif args["--tor"]:
        command = {
            "command": "--tor",
            "value": args["<value>"],
            "joint_id": args["--joint"]
        }
    elif args["--tc"]:
        command = {
            "command": "--tc",
            "joint_id": args["--joint"]
        }
    elif args["--ts"]:
        command = {
            "command": "--ts",
            "joint_id": args["--joint"]
        }
    elif args["--tptl"]:
        args["<pos>"] = args["<pos>"].replace('m','-')
        command = {
            "command": "--tptl",
            "joint_id": args["--joint"],
            "pos": args["<pos>"],
            "time": args["<time>"]
        }
    elif args["--tpts"]:
        args["<pos>"] = args["<pos>"].replace('m','-')
        command = {
            "command": "--tpts",
            "joint_id": args["--joint"],
            "pos": args["<pos>"],
            "time": args["<time>"]
        }
    elif args["--tpvt"]:
        command = {
            "command": "--tpvt",
            "csv_file": args["<csv_file>"]
        }
    else:
        print "Invalid Command"

    # Check the values
    if not isInvalidCommand(command):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.connect('tcp://localhost:5555')
        print json.dumps(command)
        sock.send(json.dumps(command))
        curState = sock.recv()
