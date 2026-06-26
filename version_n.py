import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import lmfit

from main_func import cetri_centri, sign_dati, grad_vect ,meklebias,Read_init,lorenz_find_peaks,lorenz

print("initialising") 

dat = "merged_data1.csv"
look = "Starp_lookup_table.csv"


T_range = np.linspace(-10, 200, 100)


mini = 2650
maxi = 3500
amplituda = 50
biaslauks = grad_vect(22.5,24.5)*46 
galva_bias = 20
galva = 5


def reali(testa_dati, field_lookup_df):
    freq = testa_dati["x"].values
    intens = testa_dati["y"].values
    inverted_y = -1 * intens+1

    peaks, found, lormask = lorenz_find_peaks(freq, inverted_y)

    lorenc =np.zeros_like(intens)

    Rezult = meklebias(peaks,field_lookup_df)
    

    Rez_coord = [Rezult['alfa'],Rezult['beta'],Rezult['B']]

    Rez_params = {
        "Dplus":[Rezult["Dplus1"],Rezult["Dplus2"],Rezult["Dplus3"],Rezult["Dplus4"]],
        "Dminus":[Rezult["Dminus1"],Rezult["Dminus2"],Rezult["Dminus3"],Rezult["Dminus4"]],
        "stress_dim":[Rezult['Sx'],Rezult['Sy'],0],
        "par":[Rezult['Apar1'],Rezult['Apar2'],Rezult['Apar3'],Rezult['Apar4']],
        "per":[Rezult['Aper1'],Rezult['Aper2'],Rezult['Aper3'],Rezult['Aper4']],
        "Q":[Rezult['Q1'],Rezult['Q2'],Rezult['Q3'],Rezult['Q4']]
    }

    plt.plot(freq, inverted_y, label="real")


    #plt.vlines(peaks,0,1, colors="red")
    
    print(Rez_params)
    print(f"/nrezultats(alfa,beta, bval): {Rez_coord}")
    print(f"rezultats(Bx,By,Bz): {grad_vect(Rez_coord[0],Rez_coord[1])*Rez_coord[2]}/n")
    print(f"Err: {Rezult['residual']}")


    hamfound = cetri_centri(Rez_coord,params=Rez_params, tikaienergijas = True)

    if len(hamfound) ==24: 
        for fo in range(len(found)):
            # Calculate the fit curves across the entire frequency range
            lor = lorenz(found[fo], freq)
            
            # Use the saved boolean mask to update only the specific region in 'lorenc'
            current_mask = lormask[fo]
            lorenc[current_mask] = lor[current_mask]
            found[fo]['f0'].value = hamfound[fo*3]
            found[fo]['f1'].value = hamfound[fo*3+1]
            found[fo]['f2'].value = hamfound[fo*3+2]
        
        realenz = np.zeros_like(intens)
        for fo in range(len(found)):
            # Calculate the fit curves across the entire frequency range
            lor = lorenz(found[fo], freq)
            
            # Use the saved boolean mask to update only the specific region in 'lorenc'
            current_mask = lormask[fo]
            realenz[current_mask] = lor[current_mask]
        #plt.plot(freq, lorenc, label="lorenz fit", color = "red")
        plt.plot(freq, realenz, label = "guessed", color = "green")   

    else:
        plt.plot(*cetri_centri(Rez_coord,params=Rez_params))
    plt.xlim(2820,2920)
    #plt.legend()
    plt.show()

    starp = np.array(peaks) - np.array(hamfound)

    plt.figure(figsize=(10, 6))
    plt.scatter(peaks, starp)
    plt.show()




reali(*Read_init(dat,look))

