import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr,meklebias,Read_init

coord = [30,25, 20]
ttt = 16
ds = np.linspace(-0.03, 0.03, 100)
stressdir = [-0.0097, 0.0096, 0]


odmr1, freq = cetri_centri(coord,0,stressdir,-2.16519,-2.635,-4.94588, FirstFit=True)
odmr2, freq = cetri_centri(coord,0,stressdir,-2.16519,-2.635,-4.910,FirstFit=True)

plt.plot(odmr1, freq)
plt.plot(odmr2, freq)
plt.show()