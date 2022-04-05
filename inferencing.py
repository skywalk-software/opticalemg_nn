import socket
import socketserver
import threading

from tensorflow.python.keras.models import load_model
import numpy as np
import serial
from queue import Queue

model = load_model("../models/2022_03_10_TimeseriesCNN_HL2_v0_30731/")

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

input_queue = Queue()
mean_subtracted_queue = Queue()

last_decoded_input = None
iteration_cnt = 0
click = False

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("0.0.0.0", 30785))
s.listen(1)

clients = []


def server_thread():
    print("server started!")
    while True:
        c, addr = s.accept()
        print(f"{addr} connected!")
        global clients
        clients += [c]


def send_message(message):
    disconnected_clients = []
    for client in clients:
        try:
            client.send((message + "\n").encode())
        except ConnectionResetError:
            print(f"a client has disconnected")
            disconnected_clients += [client]
    for client in disconnected_clients:
        clients.remove(client)


threading.Thread(target=server_thread, args=()).start()

while True:
    iteration_cnt += 1
    line = skywalk_serial.readline().decode('utf-8')
    # if line != "":
    #     print(line)
    state = None
    if line.startswith("o:"):
        state = 0
    elif line.startswith("x:"):
        state = 1

    if state is None:
        continue
    else:
        line = line[2:]
    if len(line.split(",")) != 20:
        continue
    decoded_input = [float(item) for item in line.split(",")]
    if state == 0:
        last_decoded_input = decoded_input
        continue
    else:
        if last_decoded_input is None:
            continue
        last_decoded_input = last_decoded_input + decoded_input[7:]

    if len(last_decoded_input) != 33:
        print(line)
        continue
    input_queue.put(last_decoded_input)
    if input_queue.qsize() <= 129:
        continue
    else:
        input_queue.get()

    aggregated_input_np = np.array(input_queue.queue)
    mean_subtracted = aggregated_input_np[-1] - np.mean(aggregated_input_np[:-1], axis=0)

    mean_subtracted_queue.put(mean_subtracted)
    if mean_subtracted_queue.qsize() < 12:
        continue
    else:
        mean_subtracted_queue.get()

    input_array = np.array([mean_subtracted_queue.queue])
    result = model(input_array)

    # print(result[0])
    if result[0][0] < result[0][1]:
        click = True
        send_message("1")
    else:
        click = False
        send_message("0")

