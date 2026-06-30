from main_func import cetri_centri,sign_dati
import numpy as np
import pandas as pd

B = np.linspace(5,30, 50)

alfas = np.linspace(20,50,30)
betas = np.linspace(0,45,45)


print("rekina koordinates")


kopkoords = []
for a in alfas:
    print(a)
    for b in betas: 
        for i in B:
            kopkoords.append([a,b,i])

print("rekina vert")

data = []

for kkord in kopkoords:

    try:
        freq, odmr = cetri_centri(kkord)
    except ValueError as e:
        print("⚠️ Tukšs all_energijas pie parametriem:")
        print("  alfa, beta, B =", kkord)
        raise  # pārtrauc izpildi, lai redzētu pirmo problēmu

    peaks, peaky = sign_dati(freq, odmr)

    if len(peaks)==24:
        kord = kkord
        starp = np.sort([peaks[23]-peaks[2],peaks[20]-peaks[5],peaks[17]-peaks[8],peaks[14]-peaks[11]])
        data.append([*kord,*starp])
        print(kkord)
    else: 
        print(f"f {kkord}") 

df = pd.DataFrame(data, columns=["alfa","beta", 'B',1,2,3,4])

print("veido csv")

df.to_csv("Starp_lookup_table.csv", index=False)
print("Done")



    