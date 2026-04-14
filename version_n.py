import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt
import time
import itertools
from scipy.optimize import least_squares

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr,peak_to_starp

T_range = np.linspace(-10, 200, 100)



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

matrix = np.array([[0, 1,1], [1, 1,1],[-1, 0,1],[0,0,1], [1, 0,1],[-1, -1,1],[0, -1,1], [1, -1,1],
              [-1, 1,0], [0, 1,0], [1, 1,0],[-1, 0,0],[0,0,0], [1, 0,0],[-1, -1,0], [0, -1,0], [1, -1,0],
              [-1, 1,-1], [0, 1,-1], [1, 1,-1],[-1, 0,-1],[0,0,-1], [1, 0,-1],[-1, -1,-1], [0, -1,-1], [1, -1,-1]])
x0 = np.array([0,0,0.03])

bounds = [
    (0, np.pi),          # theta
    (0, 2*np.pi),        # phi
    (0.0, 0.1)           # strain magnitude
]

# Read your CSV
try:
    ielasitais = pd.read_csv("Starp_lookup_table.csv")
    testa_dati = pd.read_csv("merged_data_cursed.csv")
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
values = ielasitais.iloc[:, 3:].apply(lambda row: row.dropna().tolist(), axis=1).tolist()

# Build the new DataFrame
field_lookup_df = pd.DataFrame({
    'coordinates': coords,
    'values': values
})

def Temp_lq(params, coords):
    T  = params['T']
    Sx = params['Sx']
    Sy = params['Sy']
    
    S = [Sx, Sy, 0]
    res = cetri_centri(coords, dT=T,Stress_dim_koord=S, tikaienergijas=True)
    return np.asarray(res)

def Temp_res(params, coords, real):
    pred = Temp_lq(params, coords)
    
    if len(pred) == 24:
        #print("yes")
        return real - pred
    else:
        #print(f"fuck \n {pred}")
        return np.full(24, 5000.0, dtype=float)

def Refining(p, ts):
    #print(p)
    T  = ts[0]
    Sx = ts[1]
    Sy = ts[2]
    
    coord = [p['alfa'],p['beta'],p['B']]
    
    S = [Sx, Sy, 0]
    res = cetri_centri(coord, dT=T,Stress_dim_koord=S, tikaienergijas=True)
    
    return np.asarray(res)

def Ref(p, ts, real):
    pred = Refining(p, ts)
    
    if len(pred) == 24:
        #print("yes")
        return real - pred
    else:
        #print(f"fuck \n {pred}")
        return np.full(24, 5000.0, dtype=float)


# Create parameters
params = lmfit.Parameters()
params.add('T',  value=0,   min=-10, max=100)
params.add('Sx', value=0,   min=-0.3, max=0.3)
params.add('Sy', value=0,   min=-0.3, max=0.3)

print("sākas bias meklēšana")

def meklebias(peaks):
    starppeaks = peak_to_starp(peaks)

    field_lookup_df["err"] = field_lookup_df["values"].apply(
        lambda row: np.sum(
            Vid_kvadr(np.array(row[:len(starppeaks)]), np.array(starppeaks))
        )
    )

    sorted_df = field_lookup_df.sort_values(by="err", ascending=True).reset_index(drop=True)

    vert = []
    kludinas = []

    for i in range(1,10):
        coooo = sorted_df.iloc[i].copy()
        gridmin = np.array(coooo["coordinates"])
        att = 2
        kluda = 0
        print(i)
        for a in range(10): 
            points = gridmin + matrix * att
            kludas = []
            for p in points:
                energ = sign_dati(*cetri_centri(p, tikaienergijas=False, vajagrange=True))[0]
                if len(energ) == 24:
                    starpe = peak_to_starp(energ)
                    kludas.append(np.sum(Vid_kvadr(starpe, starppeaks)))
                else:
                    kludas.append(np.inf)

            # find best point index
            best_idx = np.argmin(kludas)
            best_point = points[best_idx]
            best_error = kludas[best_idx]

            # update gridmin ONCE
            if best_error <= 10 * att:
                gridmin = best_point
                att /= 2
            else:
                print(f"apstajas pie: {abs(np.log2(att)) + 1}")
                beigu = att
                break
        vert.append(gridmin)
        kludinas.append(best_error)
    arr = np.column_stack([np.asarray(vert), np.asarray(kludinas)])  # shape (n, 4)
    # columns: [alfa, beta, B, err]

    arr_sorted = arr[np.argsort(arr[:, -1])] 

    print(np.shape(arr_sorted))
    print(arr_sorted)
    
    finall = []

    for aaa in range(5):
        rough = arr_sorted[aaa][:3]
        rez = lmfit.minimize(Temp_res, params, args=(rough, peaks),method='differential_evolution') 
        tesrt = [rez.params[name].value for name in rez.var_names]
        print (tesrt) #[22.211847912061096, -0.008420355573983851, 0.018460267821950704]

        p = lmfit.Parameters()
        p.add('alfa', value=rough[0],min= rough[0]-1,max= rough[0]+1)
        p.add('beta', value=rough[1],min= rough[1]-1,max= rough[1]+1)
        p.add('B',value=rough[2],min= rough[2]-2,max= rough[2]+2)

        end_e = lmfit.minimize(Ref, p, args=(tesrt, peaks))

        finpar = [end_e.params[name].value for name in end_e.var_names]

        # print(np.sqrt(np.sum(end_e.residual**2)))

        # print(finpar)
        finall.append([*finpar,*tesrt, np.sqrt(np.sum(end_e.residual**2))])
    finarray = np.asarray(finall)
    print(finarray)
    return finarray[finarray[:, -1].argsort()]



def letstrythisshit(alfa, beta, mag,  Print = False, graph = False):
    lauks = [alfa, beta, mag]

    lauksvec = grad_vect(alfa,beta)*mag

    freq, odmr = cetri_centri(lauks)

    peaks, peaky = sign_dati(freq, odmr)

    #print(peaks)

    reeee = meklebias(peaks)
    res = (reeee[0],reeee[1],reeee[2])

    
    paramlauks = grad_vect(res[0],res[1])*res[2]

    starp1 = []
    for i in range(3):
        starp1.append(np.abs((lauksvec[i]-paramlauks[i])*100000)) #100000


    if Print:
        print(f"īstais lauks: {lauks}")
        print(f"rezultats: {res}")
        print(f"starpiba(nT): {starp1}") 
        #print(f"Laiks: {laiks}")

    if graph:
        modfreq, mododmr = cetri_centri(res)

        plt.plot(freq, odmr, label="simulated")
        plt.plot(modfreq, mododmr, label = "guessed")
        plt.scatter(peaks, peaky)
        plt.legend()
        plt.show()

    return([[alfa, beta, mag],res, starp1])



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

    reeee = meklebias(peaks)
    res = (reeee[0],reeee[1],reeee[2])
    #peaks = [z-peakss[0] for z in peakss]
    try:
        rez,d,laiks = meklebias(peaks)
    except:
        print("fuck")
        plt.vlines(peaks,0,1, colors="red")
        plt.plot(freq, inverted_y, label="real")
        plt.xlim(2650,3100)
        plt.legend()
        plt.show()


    print(f"rezultats: {rez, d}")
    print(f"Laiks: {laiks}")

    modfreq, mododmr = cetri_centri(rez,d)
    plt.figure(figsize=(10, 6))
    
    rezfr,_ = sign_dati(modfreq, mododmr)

    plt.vlines(peaks,0,1, colors="red")
    plt.plot(freq, inverted_y, label="real")
    plt.vlines(rezfr,0,1, colors="red")
    plt.plot(modfreq, mododmr, label = "guessed")   
    plt.xlim(2650,3100)
    plt.legend()
    plt.show()

    starp = np.array(peaks) - np.array(rezfr)

    plt.figure(figsize=(10, 6))
    plt.scatter(peaks, starp)
    plt.show()

reali(testa_dati)


