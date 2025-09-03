import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from main_func import cetri_centri, sign_dati  

lenk = 30 * np.pi / 180.0
lenkmain = np.linspace(0, 2*np.pi, 10)  # better to use radians
lauks = 30


fig, ax = plt.subplots()
x0, y0 = cetri_centri([lenk, lenkmain[0], lauks])
line, = ax.plot([], [])
ax.set_xlim(np.min(x0), np.max(x0))
ax.set_ylim(np.min(y0), np.max(y0))
ax.set_aspect('auto')

def init():
    line.set_data([], [])
    return line,

def update(l):
    x, y = cetri_centri([lenk, l, lauks])
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, frames=lenkmain,
                    init_func=init, interval=200, blit=False)

plt.show()
