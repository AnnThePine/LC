import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt
import time
import itertools
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

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
    ielasitais = pd.read_csv("field_lookup_table.csv")
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
values = ielasitais.iloc[:, 4:].apply(lambda row: row.dropna().tolist(), axis=1).tolist()

skaiti = ielasitais.iloc[:, 3].astype(int).values.tolist()

# Build the new DataFrame
field_lookup_df = pd.DataFrame({
    'coordinates': coords,
    'values': values,
    'skaiti' : skaiti
})

values_by_length = {}
coords_by_length = {}

# Group rows by skaiti
for L, group in field_lookup_df.groupby("skaiti"):
    valss = np.stack(group['values'].values)          # (N, K)
    coords_arr = np.stack(group['coordinates'].values)  # (N, 3)

    values_by_length[L] = valss
    coords_by_length[L] = np.asarray(group['coordinates'].tolist(), dtype=float)

vals = values_by_length[L]      # NumPy array (N, K)
coords = coords_by_length[L]

tree_by_length = {L: KDTree(vals) for L, vals in values_by_length.items()}

def calculate_Derror(D_val, p_coord, target_peaks):
    # This is the logic you had inside your loop
    energijas = sign_dati(*cetri_centri(p_coord, D=D_val, tikaienergijas=False, vajagrange=True))[0]
    return np.sum(Vid_kvadr(energijas, target_peaks))

def calculate_stresserr(rezult, rezd,x, peaks):

    alfa,beta, P = x

    energijas = sign_dati(*cetri_centri(rezult, D=rezd,dirrr=[alfa, beta],P=P, tikaienergijas=False, vajagrange=True))[0]
    return np.sum(Vid_kvadr(energijas, peaks))

def meklebias(peaks):

    start = time.time()

    L = len(peaks)
    tree = tree_by_length.get(L)

    if tree is None:
        return None
    
    indices = tree.query_ball_point(peaks, r=np.sqrt(5000))

    if not indices:
        print("youre double fucked")
        return None
    
    dist, idx = tree.query(peaks, k=len(indices), distance_upper_bound=np.sqrt(5000))
    
    mask = np.isfinite(dist)
    resid_f = dist[mask]**2
    coords_f = coords[idx[mask]]

    if resid_f.size == 0:
        print("youre double fucked")
        return None
    
    order = np.argsort(resid_f)
    resid_sorted = resid_f[order]
    coords_sorted = coords_f[order]


    Dmekl = np.linspace(2860, 2880,20)


    #pirmais D fits
    gridmin = coords_sorted[1]
    res = minimize_scalar(calculate_Derror, args=(gridmin, peaks), bounds=(2860, 2880), method='bounded',
    options={'xatol': 0.00001} )
    datr = res.x
    
    vert = []
    kludinas = []
    
    if coords_sorted.shape[0] != 0:
        
        # mag lauka fits un šķirošana

        for i in range(1,10):
            gridmin =  coords_sorted[i].copy()
            att = 2
            kluda = 0
            print(i)
            for a in range(10): 
                points = gridmin + matrix * att
                kludas = []
                for p in points:
                    energ = sign_dati(*cetri_centri(p, D=datr, tikaienergijas=False, vajagrange=True))[0]
                    kludas.append(np.sum(Vid_kvadr(energ, peaks)))

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
        z = np.argmin(kludinas)
        rezult = vert[z]
        
        #otrais D fits 
        res = minimize_scalar(calculate_Derror, args=(rezult, peaks), bounds=(datr-1, datr+1), method='bounded',
        options={'xatol': 0.000001} )
        rezd = res.x

        #strrain virziens
        res = minimize(calculate_stresserr,
            x0=np.array(x0),
            args=(rezult, rezd, peaks),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'xtol': 1e-6,
                'ftol': 1e-6
            }
        )


        att2 = beigu*2

        gridmin = rezult
        
        for b in range(3):
            kludas = []
            matrixa = [[pair * att2  for pair in row] for row in matrix]  
            for element in matrixa:
                energijas = sign_dati(*cetri_centri([gridmin[0]+element[0],gridmin[1]+element[1],gridmin[2]+element[2]], D=rezd, tikaienergijas=False, vajagrange=True))
                kluda = np.sum(Vid_kvadr(energijas[0],peaks,square = False))
                kludas.append(kluda)
                g = kludas.index(min(kludas))
            gatrix = matrixa[g]
            gridmin = [gridmin[0]+gatrix[0],gridmin[1]+gatrix[1],gridmin[2]+element[2]]
            att2 /=2
            


        end = time.time()
        #print(f"laiks: {end - start}")
        laiks = end - start

        return gridmin,rezd, laiks
    else: 
        print(f"youre fucked\nlen peaks{len(peaks)}")
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

    rez,d,laiks = meklebias(peaks)

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


