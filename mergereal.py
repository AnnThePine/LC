import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import savgol_filter

from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr

# Folder containing your .dat files
data_folder = "data_nv_seperate"

# Find all .dat files
dat_files = glob.glob(os.path.join(data_folder, "*.dat"))

if not dat_files:
    print("No .dat files found in:", data_folder)
    exit()

merged_data = pd.DataFrame()

# --- Read and collect data from each file ---
for file_path in dat_files:
    y1 = []
    try:
        df = pd.read_csv(file_path, sep="\t", header=None)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

    x = df.iloc[:, 0]
    y1 = df.iloc[:, 2]
    
    y = (y1 - y1.min()) / (y1.max() - y1.min())
    window_size = 5
    poly_order = 2
    y_smooth = savgol_filter(y, window_size, poly_order)
    temp_df = pd.DataFrame({"x": x, "y": y_smooth})
    merged_data = pd.concat([merged_data, temp_df], ignore_index=True)

# --- Merge overlapping x-values ---
# Average all y-values for identical x
merged_data = merged_data.groupby("x", as_index=False)["y"].mean()

# --- Save to CSV ---

merged_data.to_csv("merged_data_cursed.csv", index=False)

# --- Plot merged data ---


# freq, odmr = cetri_centri([59, 19.620689655172413, 23],[0,0,1],0.03,5,-3.7,-2,4) 
# freq1, odmr1 = cetri_centri([59, 19.620689655172413, 23],[0,1,0],0.03,5,-3.7,-2,4)
plt.figure(figsize=(10, 6))
plt.plot(merged_data["x"], 1-merged_data["y"], label="Merged Data", color="blue")
# plt.plot(freq, odmr, color = "orange")
# plt.plot(freq1, odmr1, color = "green")
plt.xlabel("x (from first column)")
#plt.xlim(2800,2817)
plt.ylabel("y (from last column)")
plt.title("Merged Data from All .dat Files")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()