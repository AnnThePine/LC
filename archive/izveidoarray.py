from main_func import cetri_centri
import numpy as np
import pandas as pd

mini = 2650
maxi = 3500
step = 0.5
amplituda = 50

def generate_field_lookup(amplituda):
    lookup_data = []

    for Bx in np.arange(-amplituda, amplituda + step, step):
        print(Bx)
        for By in np.arange(-amplituda, amplituda + step, step):
            for Bz in np.arange(-amplituda, amplituda + step, step):
                model_energies = cetri_centri(
                    [Bx, By, Bz],
                    vajagrange=False,
                    tikaienergijas=True
                )
                
                model_energies = np.array(model_energies)
                model_energies = model_energies[(model_energies >= mini) & (model_energies <= maxi)]
                model_energies.sort()
                
                merged_peaks = []
                if len(model_energies) > 0:
                    current_peak = model_energies[0]
                    count = 1
                    for i in range(1, len(model_energies)):
                        if abs(model_energies[i] - current_peak) < 3.0:
                            current_peak = (current_peak * count + model_energies[i]) / (count + 1)
                            count += 1
                        else:
                            merged_peaks.append(current_peak)
                            current_peak = model_energies[i]
                            count = 1
                    merged_peaks.append(current_peak)
                
                peak_count = len(merged_peaks)
                
                # Pievienojam jaunu kolonnu ar pīķu vērtībām
                lookup_data.append([Bx, By, Bz, peak_count, merged_peaks])

    df = pd.DataFrame(lookup_data, columns=['Bx', 'By', 'Bz', 'peak_count', 'peak_values'])
    df.sort_values(by='peak_count', ascending=False, inplace=True)
    
    return df

print("Generating lookup table... This might take a while.")
field_lookup_df = generate_field_lookup(amplituda=20)
print("Lookup table generated successfully.")

field_lookup_df.to_csv("field_lookup_table.csv", index=False)
print("Lookup table saved to field_lookup_table.csv")