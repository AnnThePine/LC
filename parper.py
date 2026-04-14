import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import lmfit
from scipy.signal import find_peaks

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr

try:
    testa_dati = pd.read_csv("field_lookup_table.csv")
except FileNotFoundError:
    print("Lookup table not found. Please run the generation script first.")
    exit()

datpeaks,_ = sign_dati(testa_dati["x"],1-testa_dati["y"])

dat1 = datpeaks.iloc[0]
dat2 = datpeaks.iloc[1]
dat3 = datpeaks.iloc[2]

print(dat3-dat1)


rang = np.linspace(-30,30,20)

peak1 = []
peak2 = []
peak3 = []

#for m in pirm:
for d in rang:

        freq, odmr = cetri_centri([59.578947368421055, 19.620689655172413, 5], [1,1,1],0.03,d,0,0,0) #+_30 mpa = 0,03 gpa
        #c maina augšējo enerģ, d simetriska pret 0, noliec uz leju en., a2 does wied shit, a1 - bīda pa labi, + energijas pa kreisi

        pea = sign_dati(freq, odmr)
        peaks = pea[0]

        peak1.append(peaks[0]) 
        peak2.append(peaks[1])
        peak3.append(peaks[2]) 

data = {
    "peak1":peak1,
    "peak2":peak2,
    "peak3":peak3
}

df = pd.DataFrame(data)

df["starp1-3"] = df["peak1"] - df["peak3"]

print(df)
pirm = rang
plt.figure(figsize=(10, 6))
# plt.plot(testa_dati["x"], 1-testa_dati["y"], label="Merged Data", color="blue")
# plt.plot(freq, odmr, color = "orange")

#plt.xlim(2800,2817)
plt.title("mainot b")
plt.plot(pirm, df["peak1"] ,label = "peak1")
plt.plot(pirm, df["peak2"], label = "peak2")
plt.plot(pirm, df["peak3"], label = "peak3")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




