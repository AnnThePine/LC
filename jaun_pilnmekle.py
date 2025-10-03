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
galva = 20

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


def mekle(peaks):

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

    top = sarindots.head(galva)

    top_vert = top.iloc[:, 0]

    #return top.iloc[0, 0]

    def residual_func(p):
        model_energies = cetri_centri(
            [p['Alfa_vert'], p['Beta_vert'], p['Babs']],
            vajagrange=False,
            tikaienergijas=True
        )

        residuals = Vid_kvadr(model_energies, peaks)
        return residuals
    
    results = []
    for guess in top_vert:
        params = lmfit.Parameters()


        params.add('Alfa_vert', value=guess[0], min=0, max=70) 
        params.add('Beta_vert', value=guess[1], min=-90, max=90) #rokas nost nemainam bias
        params.add('Babs', value=guess[2], min=0, max=100)

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
    best_result = min(results, key=lambda r: r.chisqr)

    end = time.time()
    print(f"laiks: {end - start}")
    laiks = end - start

    return best_result,laiks, list(best_result.params.valuesdict().values())


def letstrythisshit(alfa, beta, mag):
    arlauks = grad_vect(alfa,beta)*mag

    koplauks = biaslauks+arlauks

    lauks = vect_grad(koplauks)

    print(f"īstais lauks: {lauks}")

    freq, odmr = cetri_centri(lauks)

    peaks, peaky = sign_dati(freq, odmr)

    res,laiks,_ = mekle(peaks)

    params = list(res.params.valuesdict().values())
    print(f"rezultats: {params}")
    starp = []
    for i in range(3):
        starp.append(lauks[i]-params[i])
    print(f"starpiba: {starp}")
    modfreq, mododmr = cetri_centri(params)

    rezlauks = (grad_vect(params[0], params[1])*params[2]) - biaslauks


    #modfreq, mododmr = cetri_centri(res)

    # plt.plot(freq, odmr, label="simulated")
    # plt.plot(modfreq, mododmr, label = "guessed")
    # plt.scatter(peaks, peaky)
    # plt.legend()
    # plt.show()

    return([[alfa, beta, mag],params, starp, laiks])


letstrythisshit(30,20, 7)
#līdz 11.2 G


# def sweep():
