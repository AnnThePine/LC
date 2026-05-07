import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr,meklebias,Read_init

print("initialising")

dat = "merged_data1.csv"
look = "Starp_lookup_table.csv"

#"merged_data_cursed.csv"

T_range = np.linspace(-10, 200, 100)


mini = 2650
maxi = 3500
amplituda = 50
biaslauks = grad_vect(22.5,24.5)*46 
galva_bias = 20
galva = 5

matrix = np.array([[0, 1,1], [1, 1,1],[-1, 0,1],[0,0,1], [1, 0,1],[-1, -1,1],[0, -1,1], [1, -1,1],
              [-1, 1,0], [0, 1,0], [1, 1,0],[-1, 0,0],[0,0,0], [1, 0,0],[-1, -1,0], [0, -1,0], [1, -1,0],
              [-1, 1,-1], [0, 1,-1], [1, 1,-1],[-1, 0,-1],[0,0,-1], [1, 0,-1],[-1, -1,-1], [0, -1,-1], [1, -1,-1]])
x0 = np.array([0,0,0.03])

bounds = [
    (0, np.pi),          # theta
    (0, 2*np.pi),        # phi
    (0.0, 0.1)           # strain magnitude
]


def reali(testa_dati, field_lookup_df):
    freq = testa_dati["x"].values
    intens = testa_dati["y"].values
    inverted_y = -1 * intens+1

#    peaks, peaky,_ = fit_lorentz_peaks(freq, inverted_y,3)
    peaks, peaky = sign_dati(freq, inverted_y)

    #peaks = [z-peakss[0] for z in peakss]
    
    reeee = meklebias(peaks,field_lookup_df)
    print(f"aaaaa{reeee}")

    res = (reeee[0],reeee[1],reeee[2])
    dT = reeee[3]
    stress_dir = [reeee[4],reeee[5],0]
    Apar = reeee[6]
    Aper=reeee[7]
    Q=reeee[8]



    plt.vlines(peaks,0,1, colors="red")
    plt.plot(freq, inverted_y, label="real")
    plt.xlim(2650,3100)
    plt.legend()



    print(f"rezultats: {res}")


    modfreq, mododmr = cetri_centri(res,dT = dT, Stress_dim_koord=stress_dir,tpar=Apar,tper=Aper,tQ=Q,FirstFit=True)

    
    rezfr,_ = sign_dati(modfreq, mododmr)

    #plt.vlines(peaks,0,1, colors="red")
    #plt.plot(freq, inverted_y, label="real")
    #plt.vlines(rezfr,0,1, colors="red")
    plt.plot(modfreq, mododmr, label = "guessed")   
    plt.xlim(2650,3100)
    plt.legend()
    plt.show()

    starp = np.array(peaks) - np.array(rezfr)

    plt.figure(figsize=(10, 6))
    plt.scatter(peaks, starp)
    plt.show()




reali(*Read_init(dat,look))

