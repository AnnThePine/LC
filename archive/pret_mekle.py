import numpy as np
import pandas as pd 
import lmfit
import matplotlib.pyplot as plt

from main_func import cetri_centri, sign_dati, grad_vect

mini = 2650
maxi = 3500
amplituda = 50

#ielādē tabulu
try:
    field_lookup_df = pd.read_csv("field_lookup_table.csv")
except FileNotFoundError:
    print("Lookup table not found. Please run the generation script first.")
    exit()

def mekle(peaks):

    target_peak_count = len(peaks)

    filtered_by_count = field_lookup_df[field_lookup_df['peak_count'] == target_peak_count].copy()


    if filtered_by_count.empty:
        print("No matching peak count found in the lookup table. Using random guesses.")
        minejumi = np.random.uniform(-amplituda, amplituda, size=(3, 100))
    else:
        # 2. solis: Aprēķinām pīķu vērtību atbilstību katram minējumam
        # `pandas` ielādē masīvu kā virkni, tāpēc mums tas ir jāpārveido
        filtered_by_count['peak_values'] = filtered_by_count['peak_values'].apply(lambda x: np.array(eval(x)))

        
        def calculate_peak_error(row):
            model_peaks = row['peak_values']
            if len(model_peaks) != len(peaks):
                return np.inf # Izslēdz minējumus ar nepareizu skaitu (lai gan tie jau ir filtrēti)

            # Aprēķina vidējo kvadrātisko atlikumu starp pīķiem
            residuals = np.sum((model_peaks - peaks)**2)
            return residuals

        filtered_by_count['error'] = filtered_by_count.apply(calculate_peak_error, axis=1)

        # 3. solis: Izvēlamies labākos minējumus ar zemāko kļūdu
        best_guesses_df = filtered_by_count.sort_values(by='error').head(100) # Ņemam top 100 labākos minējumus
        minejumi = best_guesses_df[['Bx', 'By', 'Bz']].values.T

# Definējam residual funkciju, kas ir nepieciešama lmfit
    def residual_func(p):
        model_energies = cetri_centri(
            [p['Bx'], p['By'], p['Bz']],
            vajagrange=False,
            tikaienergijas=True
        )
        model_energies = np.array(model_energies)
        model_energies = model_energies[(model_energies >= mini) & (model_energies <= maxi)]
        model_energies.sort()
        
        # apvieno tuvus pīķus
        merged_peaks = []
        if len(model_energies) > 0:
            current_peak = model_energies[0]
            count = 1
            for i in range(1, len(model_energies)):
                if abs(model_energies[i] - current_peak) < 2.0:
                    current_peak = (current_peak * count + model_energies[i]) / (count + 1)
                    count += 1
                else:
                    merged_peaks.append(current_peak)
                    current_peak = model_energies[i]
                    count = 1
            merged_peaks.append(current_peak)
        
        model_peaks = np.array(merged_peaks)
        
        # --- Ja pīķu skaits nesakrīt, dodam lielu kļūdu
        if len(model_peaks) != len(peaks):
            return np.full(len(peaks), 1e6)
        
        # --- Ja skaits sakrīt: sakārtojam un salīdzinām 1:1
        model_peaks_sorted = np.sort(model_peaks)
        peaks_sorted = np.sort(peaks)
        
        residuals = model_peaks_sorted - peaks_sorted
        return residuals


    results = []
    for guess in minejumi.T:  # paņemam 5 labākos
        params = lmfit.Parameters()
        params.add('Bx', value=guess[0], min=-amplituda, max=amplituda)
        params.add('By', value=guess[1], min=-amplituda, max=amplituda)
        params.add('Bz', value=guess[2], min=-amplituda, max=amplituda)

        result = lmfit.minimize(
            residual_func, 
            params,
            method="leastsq",
            ftol=1e-10, xtol=1e-10
        )
        results.append(result)

    # izvēlamies labāko
    best_result = min(results, key=lambda r: r.chisqr)
    return best_result, list(best_result.params.valuesdict().values())


def letstrythisshit(alfa, beta, mag):
    #lauks = grad_vect(alfa, beta)*mag #g
    lauks = [6.0,19.5,9.5]
    print(lauks)
    freq, odmr = cetri_centri(lauks)

    peaks, peaky = sign_dati(freq, odmr)

    res,_ = mekle(peaks)

    params = list(res.params.valuesdict().values())
    modfreq, mododmr = cetri_centri(params)
    print(params)

    plt.plot(freq, odmr, label="simulated")
    plt.plot(modfreq, mododmr, label = "guessed")
    plt.scatter(peaks, peaky)
    plt.legend()
    plt.show()

letstrythisshit(30,20, 30)


