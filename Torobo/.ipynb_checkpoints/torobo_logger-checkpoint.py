import numpy as np
from time import sleep, time
from torobo_communicator import request_state
import json


freq = 25.0
history = []
base_time = time()


def main():
    try:
        while True:
            state_json = json.loads(request_state())
            current_time = time() - base_time
            pos = [js['position'] for js in state_json['jointState']]
            history.append([current_time] + pos)
            sleep(1.0 / freq)
    except KeyboardInterrupt:
        np.save('history', np.array(history))


if __name__ == '__main__':
    main()
