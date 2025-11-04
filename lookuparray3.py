from main_func import cetri_centri,sign_dati
import numpy as np
import pandas as pd



B = np.linspace(15,60, 20)

alfas = np.linspace(20,75,20)
betas = np.linspace(0,90,30)

agper = np.linspace(-9,-5,5)
# agper = -7
agpar = np.linspace(-2,0,5)
# agpar = -1.7

print("rekina koordinates")


kopkoords = []
for a in alfas:
    print(a)
    for b in betas: 
        for i in B:
            for pe in agper:
                for par in agpar:
                    kopkoords.append([np.array([a,b,i]),pe,par])

print("rekina vert")

data = []

for kkord in kopkoords:
    print(kkord)
    try:
        freq, odmr = cetri_centri(*kkord)
    except ValueError as e:
        print("⚠️ Tukšs all_energijas pie parametriem:")
        print("  alfa, beta, B =", kkord[0])
        print("  agper, agpar  =", kkord[1], kkord[2])
        raise  # pārtrauc izpildi, lai redzētu pirmo problēmu
    peaks, peaky = sign_dati(freq, odmr)
    kord = kkord[0]
    data.append([*kord,kkord[1],kkord[2], len(peaks), *peaks])

df = pd.DataFrame(data, columns=["alfa","beta", 'B',"agper","agparal", "skaits",1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

print("veido csv")

df.to_csv("field_lookup_table.csv", index=False)
print("Done")



    