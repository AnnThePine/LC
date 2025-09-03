import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from main_func import cetri_centri, sign_dati


freq, odmr = cetri_centri((-25,22.5,46),tikaienergijas = False,vajagrange = True)


plt.plot(freq, odmr)
plt.show()