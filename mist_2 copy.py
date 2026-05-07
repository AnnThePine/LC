import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt

from main_func import cetri_centri, sign_dati, grad_vect, ipasvertibas,Stress_tensor,Read_init,Temperature_dependance

coord = [30,25, 20]
ttt = 16
ds = np.linspace(-0.03, 0.03, 100)
stressdir = [-0.0097, 0.0096, 0]
Mx, My, Mz = Stress_tensor(*stressdir)

D,Q,Apar,Aper = Temperature_dependance(50)

odmr1 = ipasvertibas(29.232237655561452,62.99367378128563, 16,2870,Mx, My, Mz,Apar,Aper,Q)
#odmr2 =  ipasvertibas(29.232237655561452,62.99367378128563, 16,2870,Mx, My, Mz,-2.16519,-2.635,-4.2)
odmr3 =  ipasvertibas(29.232237655561452,62.99367378128563, 16,2867,Mx, My, Mz,-2.16519,0,-4.94588)
odmr4 =  ipasvertibas(29.232237655561452,62.99367378128563, 16,2870,Mx, My, Mz,Apar,Aper,Q)

odmr2 = [odmr1[8]-odmr1[2],odmr1[5]-odmr1[2],
         odmr1[7]-odmr1[1],odmr1[3]-odmr1[1],
         odmr1[6]-odmr1[0],odmr1[4]-odmr1[0]]

dat = "merged_data1.csv"
look = "Starp_lookup_table.csv"

testa_dati,_= Read_init(dat,look)

freq = testa_dati["x"].values
intens = testa_dati["y"].values
inverted_y = -1 * intens+1

plt.plot(freq, inverted_y, label="real",color='orange')

#plt.vlines(odmr1,0,1,color='red')
plt.vlines(odmr2,0,1,color='green')
#plt.vlines(odmr3,0,1,color='purple')
#plt.vlines(odmr4,0,1,color='blue')
plt.xlim(2800,2925)
plt.show()