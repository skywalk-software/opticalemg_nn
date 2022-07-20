import socket
import threading
import time

import keyboard as keyboard

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
            print(message)
            client.send((message + "\n").encode())
        except ConnectionResetError:
            print(f"a client has disconnected")
            disconnected_clients += [client]
    for client in disconnected_clients:
        clients.remove(client)


threading.Thread(target=server_thread, args=()).start()

while True:
    time.sleep(0.01)
    if keyboard.is_pressed('f11'):
        click = True
        print("click")
        send_message("1")
    else:
        click = False
        print("no click")
        send_message("0")
