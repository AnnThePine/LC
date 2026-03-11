import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from scipy.signal import savgol_filter

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr

try:
    testa_dati = pd.read_csv("merged_data_cursed.csv")
except FileNotFoundError:
    print("Lookup table not found. Please run the generation script first.")
    exit()
y1 = 1-testa_dati["y"].values
y = y1/max(y1)
# window_size = 4
# poly_order = 3
# y_smooth = savgol_filter(y, window_size, poly_order)


x = testa_dati["x"].values

peaks, peaky = sign_dati(x,y)


plt.plot(x,y)
plt.scatter(peaks, peaky)
plt.show()