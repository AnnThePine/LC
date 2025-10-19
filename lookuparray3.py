from main_func import cetri_centri,sign_dati
import numpy as np
import pandas as pd

skaits = 100
center_x, center_y = 22.5, 24.5
radius = 13


B = np.linspace(10,80, 35)

alfas = np.linspace(0,90,skaits)
betas = np.linspace(0,90,skaits)

print("rekina koordinates")


koords = []
for a in alfas:
    for b in betas: 
        for i in B:
            koords.append(np.array([a,b,i]))


print("rekina vert")

data = []

for kord in koords:
    freq, odmr = cetri_centri(kord)
    peaks, peaky = sign_dati(freq, odmr)
    data.append([*kord, len(peaks), *peaks])

df = pd.DataFrame(data, columns=["alfa","beta", 'B', "skaits",1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

print("veido csv")

df.to_csv("field_lookup_table.csv", index=False)
print("Done")



    