import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial
import matplotlib

matplotlib.use('macosx')
# %%
fig, ax = plt.subplots()
lines = [ax.plot(np.random.rand(10), label=f"{i}")[0] for i in range(12)]
# ax.set_ylim(-5000, 5000)
xdata, ydata = [0] * 90, [[0 for i in range(12)] for j in range(90)]
SERIAL_PORT = "/dev/cu.usbmodem82125401"
SERIAL_BAUD_RATE = 115200
raw = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD_RATE)


# %%

def run(data):
    t, y = data
    del xdata[0]
    del ydata[0]
    xdata.append(t)
    ydata.append(y)
    ydata_np = np.array(ydata)
    ax.set_ylim(0, 100000)
    ax.set_xlim(min(xdata), max(xdata))
    for i in range(12):
        lines[i].set_data(xdata, ydata_np[:, i])
    return lines


def data_gen():
    t = 0
    while True:
        line = raw.readline()
        line = raw.readline()
        line = raw.readline()
        line = raw.readline()
        line = raw.readline()
        decoded_input = [float(item) for item in line.decode('utf-8').split(",")[:-1]]
        # print("called")
        if len(decoded_input) != 12:
            print(decoded_input)
            continue
        t += 0.1
        yield t, decoded_input
    print("exited!")

#%%

ani = animation.FuncAnimation(fig, run, frames=data_gen, interval=1, blit=False)
plt.show()


#%%

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')
#
# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return ln,
#
# def update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     ln.set_data(xdata, ydata)
#     return ln,
#
# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128), interval=20, init_func=init, blit=True)
# plt.show(block = False)