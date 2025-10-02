from main_func import cetri_centri
import numpy as np
import pandas as pd

skaits = 26
center_x, center_y = 22.5, 24.5
radius = 13


B = np.linspace(52.8,75.2, 20)

alfas = np.linspace(center_x-radius,center_x+radius,skaits)
betas = np.linspace(center_y-radius,center_y+radius,skaits)

print("rekina koordinates")

lenki = [(a, b) for a in alfas for b in betas if (a - center_x)**2 + (b - center_y)**2 <= radius**2]

koords = []
for a,b in lenki:
    for i in B:
        koords.append(np.array([a,b,i]))


print("rekina vert")

data = []

for kord in koords:
    vert = cetri_centri(kord, tikaienergijas=True)
    data.append([*kord, len(vert), *vert])

df = pd.DataFrame(data, columns=["alfa","beta", 'B', "skaits",1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])

print("veido csv")

df.to_csv("field_lookup_table.csv", index=False)
print("Done")



    






    

# def generate_field_lookup(amplituda):
#     lookup_data = []

#     for Bx in np.arange(-amplituda, amplituda + step, step):
#         print(Bx)
#         for By in np.arange(-amplituda, amplituda + step, step):
#             for Bz in np.arange(-amplituda, amplituda + step, step):
#                 model_energies = cetri_centri(
#                     [Bx, By, Bz],
#                     vajagrange=False,
#                     tikaienergijas=True
#                 )
                
#                 model_energies = np.array(model_energies)
#                 model_energies = model_energies[(model_energies >= mini) & (model_energies <= maxi)]
#                 model_energies.sort()
                
#                 merged_peaks = []
#                 if len(model_energies) > 0:
#                     current_peak = model_energies[0]
#                     count = 1
#                     for i in range(1, len(model_energies)):
#                         if abs(model_energies[i] - current_peak) < 3.0:
#                             current_peak = (current_peak * count + model_energies[i]) / (count + 1)
#                             count += 1
#                         else:
#                             merged_peaks.append(current_peak)
#                             current_peak = model_energies[i]
#                             count = 1
#                     merged_peaks.append(current_peak)
                
#                 peak_count = len(merged_peaks)
                
#                 # Pievienojam jaunu kolonnu ar pīķu vērtībām
#                 lookup_data.append([Bx, By, Bz, peak_count, merged_peaks])

#     df = pd.DataFrame(lookup_data, columns=['Bx', 'By', 'Bz', 'peak_count', 'peak_values'])
#     df.sort_values(by='peak_count', ascending=False, inplace=True)
    
#     return df

# print("Generating lookup table... This might take a while.")
# field_lookup_df = generate_field_lookup(amplituda=20)
# print("Lookup table generated successfully.")

# field_lookup_df.to_csv("field_lookup_table.csv", index=False)
# print("Lookup table saved to field_lookup_table.csv")