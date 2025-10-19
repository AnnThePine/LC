import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from main_func import cetri_centri, sign_dati


freq, odmr = cetri_centri((16,20,10),tikaienergijas = False,vajagrange = True)
peaks, peaky = sign_dati(freq, odmr)

plt.plot(freq, odmr)
plt.scatter(peaks, peaky)
plt.show()