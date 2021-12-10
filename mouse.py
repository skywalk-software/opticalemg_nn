from model import NnModel
from queue import Queue

import numpy as np
import serial
import pyautogui
import torch

mod = NnModel.load_from_checkpoint("../models/nn2_5_1.ckpt", learning_rate=0.001)

pyautogui.FAILSAFE = False

arr = [0, 0, 0, 0]
aggregated_input = []
screenWidth, screenHeight = pyautogui.size()
SERIAL_PORT = "/dev/tty.usbmodem01234567891"
SERIAL_BAUD_RATE = 115200
skywalk_serial = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD_RATE)
# read until timeout
skywalk_serial.timeout = 0
while True:
    try:
        line = skywalk_serial.readline()
        if line == b'':
            break
    except serial.SerialTimeoutException:
        break
iter = 0
# while iter < 1000:
#     line = skywalk_serial.readline().decode('utf-8')
#     if not line.startswith("o:"):
#         continue
#     else:
#         line = line[2:]
#     decoded_input = [float(item) / 10000 for item in line.split(",")]
#     iter += 1

# resume unlimited timeout
skywalk_serial.timeout = None

input_queue = Queue()

# process skywalk data
iter = 0
while True:
    iter += 1
    line = skywalk_serial.readline().decode('utf-8')
    if not line.startswith("o:"):
        continue
    else:
        line = line[2:]
    decoded_input = [float(item) / 10000 for item in line.split(",")]
    if len(decoded_input) != 20:
        print(line)
        continue
    input_queue.put(decoded_input)
    if input_queue.qsize() <= 64:
        continue
    else:
        input_queue.get()
    # global aggregated_input, calibrated, skywalk_min, skywalk_max
    # aggregated_input += [decoded_input]
    # if not calibrated:
    #     print(len(aggregated_input))
    #     if len(aggregated_input) < 300:
    #         continue
    #     else:
    #         aggregated_input_np = np.array(aggregated_input)
    #         skywalk_min = np.min(aggregated_input_np, axis=0)
    #         skywalk_max = np.max(aggregated_input_np, axis=0)
    #         calibrated = True
    #         aggregated_input = []

    # if len(aggregated_input) != 90:
    #     continue
    aggregated_input_np = np.array(input_queue.queue)
    if iter % 30 != 1:
        continue
    # for i in range(aggregated_input_np.shape[1]):
    #     aggregated_input_np[:, i] = normalize(aggregated_input_np[:, i], skywalk_min[i], skywalk_max[i])
    input_tensor = torch.FloatTensor([aggregated_input_np])
    output_tensor = mod(input_tensor)
    # print("input_tensor", input_tensor)
    # print("output_tensor", output_tensor)
    aggregated_input = aggregated_input[1:]
    # print(f"{output_tensor[0, 0]: 0.3f} \t {output_tensor[0, 1]: 0.3f}")
    arr[0] = output_tensor[0, 0].item()
    arr[1] = output_tensor[0, 1].item()
    arr[2] = output_tensor[0, 2].item() / 2 + 2
    arr[3] = output_tensor[0, 3].item() / 2 + 2
    x = (arr[0] - (-0.3)) / (0.5 - (-0.3))
    y = (arr[1] - (-0.5)) / ((0.4) - (-0.5))
    # pyautogui.moveTo(((np.clip(y, 0, 1)) * screenWidth), (1 - np.clip(x, 0, 1)) * screenHeight)
    # x = arr[0]
    # y = arr[1]
    print(x, y)
