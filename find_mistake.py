import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr,meklebias

coord = [30,25, 20]
ttt = 16
ds = np.linspace(-0.03, 0.03, 100)
stressdir = [-0.0097, 0.0096, 0]

max_levels = 24
E = np.full((len(ds), max_levels), np.nan)

for k, sw in enumerate(ds):
    en = np.asarray(
        cetri_centri(coord, dT=ttt, Stress_dim_koord=stressdir,
                     tikaienergijas=True, FirstFit=True,tper=sw)
    )
    n = min(en.size, max_levels)
    E[k, :n] = en[:n]   # if some are missing, rest stays NaN

for i in range(max_levels):
    mask = ~np.isnan(E[:, i])
    if np.any(mask):
        plt.plot(ds[mask], E[mask, i])

# plt.plot(*cetri_centri([30,25, 25]))
# plt.plot(*cetri_centri([30,25, 25]))

# plt.xlabel("Pa")
# plt.ylabel("MHz")
# plt.title("change x strain (missing levels -> gaps)")
plt.show()