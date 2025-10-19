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
values = ielasitais.iloc[:, 4:].values.tolist()

skaiti = ielasitais.iloc[:, 3].values.tolist()

# Build the new DataFrame
field_lookup_df = pd.DataFrame({
    'coordinates': coords,
    'values': values
})


def meklebias(peaks, minimalaisalfa, maximalaisalfa,minimalaisbeta ,maximalaisbeta,minatt):

    start = time.time()

    df = field_lookup_df

    resid = []
    for i in range(len(df)):
        if len(peaks)== skaiti[i]:
            resid.append(np.sum(Vid_kvadr(df.iloc[i, 1],peaks)))
        else:
            resid.append(np.inf)
    
    df.loc[:,'vid kludas'] = resid 


    sarindots = df.sort_values('vid kludas', ascending=True)

    top = sarindots.head(galva_bias)

    top_vert = top.iloc[:, 0]

    #return top.iloc[0, 0]

    def residual_func(p):
        model_energies = cetri_centri(
            [p['Alfa_vert'], p['Beta_vert'], p['Babs']],
            vajagrange=False,
            tikaienergijas=True
        )

        # ja nepareizs garums vai NaN → sods
        if (len(model_energies) != len(peaks)) or np.any(np.isnan(model_energies)):
            return np.ones(len(peaks)) * 1e9

        residuals = Vid_kvadr(model_energies, peaks)*100000

        if np.any(np.isnan(residuals)):
            return np.ones(len(peaks)) * 1e9

        return residuals
    
    results = []
    for guess in top_vert:
        params = lmfit.Parameters()


        #params.add('Alfa_vert', value=guess[0], min=minimalaisalfa, max=maximalaisalfa) 
        params.add('Alfa_vert', value=guess[0], min=10, max=70) 
        #params.add('Beta_vert', value=guess[1], min= minimalaisbeta, max=maximalaisbeta) #rokas nost nemainam iekš bias
        params.add('Beta_vert', value=guess[1], min= -90, max=90) #rokas nost nemainam iekš bias
        params.add('Babs', value=guess[2], min=15, max=25)

        result = lmfit.minimize(
            residual_func,
            params,
            method="least_squares",      # uses scipy.optimize.least_squares
            loss='linear',               # pure least squares
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=20000               # allow more evaluations
        )
        results.append(result)


    # izvēlamies labāko
    sorted_results = sorted(results, key=lambda r: r.chisqr)

    end = []
    for i in range(5):

        best = list(sorted_results[i].params.valuesdict().values())

        params = lmfit.Parameters()

        params.add('Alfa_vert', value=best[0], min=best[0]-minatt, max=best[0]+minatt) 
        params.add('Beta_vert', value=best[1], min=best[1]-minatt, max=best[1]+minatt) #rokas nost nemainam iekš bias
        params.add('Babs', value=best[2], min=best[2]-minatt, max=best[2]+minatt)

        result = lmfit.minimize(
            residual_func,
            params,
            method="least_squares",      # uses scipy.optimize.least_squares
            loss='linear',               # pure least squares
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=20000               # allow more evaluations
        )

        end.append(result)

    best_result = min(end, key=lambda r: r.chisqr)
        



    end = time.time()
    print(f"laiks: {end - start}")
    laiks = end - start

    return result,laiks, list(best_result.params.valuesdict().values())


def letstrythisshit(alfa, beta, mag,minimalaisalfa, maximalaisalfa, minimalaisbeta ,maximalaisbeta,minatt,  Print = False):
    arlauks = grad_vect(alfa,beta)*mag

    koplauks = biaslauks+arlauks

    lauks = vect_grad(arlauks)

    freq, odmr = cetri_centri(lauks)

    peaks, peaky = sign_dati(freq, odmr)

    print(peaks)

    res,laiks,_ = meklebias(peaks, minimalaisalfa, maximalaisalfa, minimalaisbeta ,maximalaisbeta,minatt)

    params = list(res.params.valuesdict().values())

    paramlauks = grad_vect(params[0],params[1])*params[2]
    starp1 = []
    for i in range(3):
        starp1.append(np.abs((arlauks[i]-paramlauks[i]))) #100000

    rezlauks = (grad_vect(params[0], params[1])*params[2]) - biaslauks


    if Print:
        print(f"īstais lauks: {lauks}")
        print(f"rezultats: {params}")
        print(f"starpiba: {starp1}")

    modfreq, mododmr = cetri_centri(params)

    plt.plot(freq, odmr, label="simulated")
    plt.plot(modfreq, mododmr, label = "guessed")
    plt.scatter(peaks, peaky)
    plt.legend()
    plt.show()

    return([[alfa, beta, mag],params, starp1, laiks])


def sweep(att,minatt,graf = False):
    alfas = np.linspace(10, 50, 40)

    starpibas1 = []
    starpibas2 = []
    starpibas3 = []
    laiki = []
    for beta in alfas:
        _,_,starp,laiks = letstrythisshit(beta, 50, 20, beta-att, beta+att, 0,90, minatt)
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


def kludaas():
    pirmie = np.linspace(1,10,15)
    otrie = np.linspace(0.5,20,20)

    klud1 = []
    klud2 = []
    klud3 = []
    pirmais = 3
    #for pirmais in pirmie:
    for otrais in otrie:
        k1,k2,k3 = sweep(pirmais,otrais)
        klud1.append(k1)
        klud2.append(k2)
        klud3.append(k3)

    plt.plot(otrie,klud1)
    plt.plot(otrie,klud2)
    plt.plot(otrie,klud3)
    plt.show()

kludaas()


