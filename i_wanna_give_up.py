import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt
import time
import itertools

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr

mini = 2650
maxi = 3500
amplituda = 50
biaslauks = grad_vect(22.5,24.5)*46 
galva_bias = 20
galva = 5

minalfa = 5
maxalfa = 85

minB = 5
maxB = 500




# Read your CSV
try:
    ielasitais = pd.read_csv("field_lookup_table1.csv")
    testa_dati = pd.read_csv("merged_data0.csv")
except FileNotFoundError:
    print("Lookup table not found. Please run the generation script first.")
    exit()


# Apply both filters in one vectorized operation
ielasitais = ielasitais[
    (ielasitais['alfa'] >= minalfa) & (ielasitais['alfa'] <= maxalfa) &
    (ielasitais['B'] >= minB) & (ielasitais['B'] <= maxB)
].reset_index(drop=True)

# Combine the first three columns into a tuple or string
# Example: tuple
coords = list(zip(
    ielasitais.iloc[:, 0],  # alfa
    ielasitais.iloc[:, 1],  # beta
    ielasitais.iloc[:, 2]   # B
))

# Collect the rest of the columns into lists
values = ielasitais.iloc[:, 6:].apply(lambda row: row.dropna().tolist(), axis=1).tolist()

skaiti = ielasitais.iloc[:, 5].astype(int).values.tolist()

agpar = ielasitais.iloc[:, 4].astype(int).values.tolist()

agper = ielasitais.iloc[:, 3].astype(int).values.tolist()

# Build the new DataFrame
field_lookup_df = pd.DataFrame({
    'coordinates': coords,
    "agpar": agpar,
    "agper":agper,
    'values': values,
    'skaiti' : skaiti
})

def meklebias(peaks):

    start = time.time()

    df = field_lookup_df[field_lookup_df['skaiti'] == len(peaks)].reset_index(drop=True)

    

    resid = [np.sum(Vid_kvadr(row, peaks)) for row in df['values']]
    df['vid kludas'] = resid
    
    so = df[df['vid kludas'] <= 50000]
    
    top = so.sort_values('vid kludas', ascending=True)
    
    
    if len(top) >= 10:
        top = top.head()

    top_vert = top.iloc[:, 0]
    toppar = top.iloc[:, 1]
    topper = top.iloc[:, 2]

    matrix = [[-1, 1,1], [0, 1,1], [1, 1,1],[-1, 0,1],[0,0,1], [1, 0,1],[-1, -1,1],[0, -1,1], [1, -1,1],
              [-1, 1,0], [0, 1,0], [1, 1,0],[-1, 0,0],[0,0,0], [1, 0,0],[-1, -1,0], [0, -1,0], [1, -1,0],
              [-1, 1,-1], [0, 1,-1], [1, 1,-1],[-1, 0,-1],[0,0,-1], [1, 0,-1],[-1, -1,-1], [0, -1,-1], [1, -1,-1]]
    
    listp = [-1,0,1]

    matrix5D = [list((*xyz, par, per)) for xyz, par, per in itertools.product(matrix, listp, listp)]

    

    vert = []
    kludinas = []
    
    if len(top_vert) != 0:

        for i in range(len(top_vert)):
            gridmin = top_vert.iloc[i]
            parmin=toppar.iloc[i]
            permin=topper.iloc[i]
            
            att = 4
            kluda = 0
            print(i)
            for a in range(10): 
                matrixa = [[pair * att  for pair in row] for row in matrix5D]  
                kludas = []
                for element in matrixa:
                            energijas = sign_dati(*cetri_centri([gridmin[0]+element[0],gridmin[1]+element[1],gridmin[2]+element[2]],permin+element[3],parmin+element[4], tikaienergijas=False, vajagrange=True))
                            kluda = np.sum(Vid_kvadr(energijas[0],peaks))
                            kludas.append(kluda)
                            g = kludas.index(min(kludas))
                gatrix = matrixa[g]
                if min(kludas) <= 80*att:
                    gridmin = [gridmin[0]+gatrix[0],gridmin[1]+gatrix[1],gridmin[2]+gatrix[2]]
                    parmin = parmin+gatrix[4]
                    permin = permin+gatrix[3]
                    att /=2
                else: 
                    print(f"apstajas pie: {abs(np.log2(att))+1}")
                    beigu = att
                    gridmin = [gridmin[0]+gatrix[0],gridmin[1]+gatrix[1],gridmin[2]+gatrix[2]]
                    parmin = parmin+gatrix[4]
                    permin = permin+gatrix[3]
                    break
            vert.append([gridmin,parmin,permin])
            kludinas.append(min(kludas))
        z = kludinas.index(min(kludinas))
        rezult = vert[z]
        

        att2 = beigu*16

        # gridmin = rezult
        
        # for b in range(5):
        #     kludas = []
        #     matrixa = [[pair * att2  for pair in row] for row in matrix]  
        #     for element in matrixa:
        #         energijas = sign_dati(*cetri_centri([gridmin[0]+element[0],gridmin[1]+element[1],gridmin[2]+element[2]], tikaienergijas=False, vajagrange=True))
        #         kluda = np.sum(Vid_kvadr(energijas[0],peaks))
        #         kludas.append(kluda)
        #     g = kludas.index(min(kludas))
        #     gatrix = matrixa[g]
        #     gridmin = [gridmin[0]+gatrix[0],gridmin[1]+gatrix[1],gridmin[2]+element[2]]
        #     att2 /=2
            


        end = time.time()
        #print(f"laiks: {end - start}")
        laiks = end - start

        return rezult, laiks
    else: 
        print(f"youre fucked\n{df}\nlen peaks{len(peaks)}")
        end = time.time()
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
    alfas = np.linspace(minalfa, maxalfa, 40)

    starpibas1 = []
    starpibas2 = []
    starpibas3 = []
    laiki = []
    for beta in alfas:
        _,_,starp,laiks = letstrythisshit(beta,25,30, Print=True, graph = False)
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
#letstrythisshit(13,25,45, Print=True, graph = True)

def reali(testa_dati):
    freq = testa_dati["x"].values
    intens = testa_dati["y"].values
    inverted_y = -1 * intens+1

    peaks, peaky = sign_dati(freq, inverted_y)

    rez,laiks = meklebias(peaks)
    res = rez[0]


    print(f"rezultats: {rez}")
    print(f"Laiks: {laiks}")

    modfreq, mododmr = cetri_centri(*rez)
    plt.figure(figsize=(10, 6))
    plt.plot(freq, inverted_y, label="real")
    plt.plot(modfreq, mododmr, label = "guessed")
    plt.scatter(peaks, peaky)
    plt.legend()
    plt.show()

reali(testa_dati)


