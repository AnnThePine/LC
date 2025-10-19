import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt
import time

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr

mini = 2650
maxi = 3500
amplituda = 50
biaslauks = grad_vect(22.5,24.5)*46 
galva_bias = 20
galva = 5



# Read your CSV
try:
    ielasitais = pd.read_csv("field_lookup_table.csv")
except FileNotFoundError:
    print("Lookup table not found. Please run the generation script first.")
    exit()

# Combine the first three columns into a tuple or string
# Example: tuple
coords = list(zip(
    ielasitais.iloc[:, 0],  # alfa
    ielasitais.iloc[:, 1],  # beta
    ielasitais.iloc[:, 2]   # B
))

# Collect the rest of the columns into lists
values = ielasitais.iloc[:, 4:].apply(lambda row: row.dropna().tolist(), axis=1).tolist()

skaiti = ielasitais.iloc[:, 3].astype(int).values.tolist()

# Build the new DataFrame
field_lookup_df = pd.DataFrame({
    'coordinates': coords,
    'values': values,
    'skaiti' : skaiti
})


def meklebias(peaks):

    start = time.time()
    
    df = field_lookup_df

    resid = []
    for i in range(len(df)):
        if len(peaks) == df.iloc[i, 2]:
            resid.append(np.sum(Vid_kvadr(df.iloc[i, 1],peaks)))
        else:
            resid.append(np.inf)
    
    df.loc[:,'vid kludas'] = resid 

    so = df[df['vid kludas'] <= 2]

    top = so.sort_values('vid kludas', ascending=True)

    top_vert = top.iloc[:, 0]
    
    if len(top_vert) >= 10:
        top_vert = top_vert.head()

    matrix = [[-1, 1], [0, 1], [1, 1],[-1, 0],[0,0], [1, 0],[-1, -1], [0, -1], [1, -1]]

    vert = []
    kludinas = []
    
    if len(top_vert) != 0:

        for i in range(len(top_vert)):
            gridmin = top_vert.iloc[i]
            att = 2
            kluda = 0
            print(i)
            for a in range(5): 
                matrixa = [[pair * att  for pair in row] for row in matrix]  
                kludas = []
                for element in matrixa:
                    energijas = sign_dati(*cetri_centri([gridmin[0]+element[0],gridmin[1]+element[1],gridmin[2]], tikaienergijas=False, vajagrange=True))
                    kluda = np.sum(Vid_kvadr(energijas[0],peaks))
                    kludas.append(kluda)
                g = kludas.index(min(kludas))
                gatrix = matrixa[g]
                gridmin = [gridmin[0]+gatrix[0],gridmin[1]+gatrix[1],gridmin[2]]
                if min(kludas) <= att:
                    att /=2
                else: 
                    print("aaaa")
                    break
            vert.append(gridmin)
            kludinas.append(min(kludas))
        z = kludinas.index(min(kludinas))
        rezult = vert[z]
        #print(rezult)


        end = time.time()
        #print(f"laiks: {end - start}")
        laiks = end - start

        return rezult, laiks
    else: 
        print(f"youre fucked\n{df}")
        end = time.time()
        #print(f"laiks: {end - start}")
        laiks = end - start
        return[0,0,0],laiks

def letstrythisshit(alfa, beta, mag,  Print = False, graph = False):
    lauks = [alfa, beta, mag]

    lauksvec = grad_vect(alfa,beta)*mag

    freq, odmr = cetri_centri(lauks)

    peaks, peaky = sign_dati(freq, odmr)

    #print(peaks)

    res,laiks = meklebias(peaks)

    
    paramlauks = grad_vect(res[0],res[1])*res[2]

    starp1 = []
    for i in range(3):
        starp1.append(np.abs((lauksvec[i]-paramlauks[i])*100000)) #100000


    if Print:
        print(f"īstais lauks: {lauks}")
        print(f"rezultats: {res}")
        print(f"starpiba(nT): {starp1}") 
        print(f"Laiks: {laiks}")

    if graph:
        modfreq, mododmr = cetri_centri(res)

        plt.plot(freq, odmr, label="simulated")
        plt.plot(modfreq, mododmr, label = "guessed")
        plt.scatter(peaks, peaky)
        plt.legend()
        plt.show()

    return([[alfa, beta, mag],res, starp1, laiks])





def sweep(graf = False):
    alfas = np.linspace(10, 50, 40)

    starpibas1 = []
    starpibas2 = []
    starpibas3 = []
    laiki = []
    for beta in alfas:
        _,_,starp,laiks = letstrythisshit(beta,25,45, Print=True, graph = False)
        starpibas1.append(starp[0])
        starpibas2.append(starp[1])
        starpibas3.append(starp[2])
        #print(starpibas1)
        laiki.append(laiks)

    if graf:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(alfas, starpibas1, label = "x")
        ax1.plot(alfas, starpibas2, label = "y")
        ax1.plot(alfas, starpibas3, label = "z")
        ax1.legend()
        ax1.set_xlabel("alfa leņķis, grad")
        ax1.set_ylabel("mag. lauka kļūdas nanoteslās")
        ax2.plot(alfas, laiki)
        ax2.set_xlabel("alfa leņķis, grad")
        ax2.set_ylabel('laiks, sec')

        plt.tight_layout()
        plt.show()
    
    return(max(starpibas1),max(starpibas2),max(starpibas3))

#sweep(graf =True)
letstrythisshit(16,20,10, Print=True, graph = True)