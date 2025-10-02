import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main_func import cetri_centri, sign_dati

# --- Parameters ---
ik_pec_lenkis = 2  # Step size for alpha (latitude) in degrees
tol = 1 # Tolerance for merging close peaks

# Angle range limits
alpha_min = -45
alpha_max = 0
beta_min = 0
beta_max = 90

# Values for lauks (sphere radius) to test
lauks_values = [32, 64, 128]
area_results = []
area_24_only_results = []

# --- Helper Function ---
def merge_close_peaks(energies):


    if len(energies) == 0:
        return []
    energies = np.sort(energies) # Sort ascending
    merged = [energies[0]]
    for e in energies[1:]:
        if abs(e - merged[-1]) <= tol:
            merged[-1] = (merged[-1] + e) / 2
        else:
            merged.append(e)
    return merged

# --- Main Processing Loop ---
# Loop through different lauks values
for lauks in lauks_values:
    total_area = 0.0
    total_area_24_only = 0.0
    vert = {
        "24": [],
        "23": [],
        "22-21": [],
        "20-15": [],
        "15-10": [],
        "under 10": []
    } 

# Iterate through alpha (latitude) with a constant step.
    for alpha in range(alpha_min, alpha_max, ik_pec_lenkis):
 # Calculate the number of beta points needed for this latitude band.
        n_beta = max(1, int((beta_max - beta_min) / ik_pec_lenkis * np.cos(np.deg2rad(alpha))))

 # Create the beta points for this specific latitude band.
        for beta in np.linspace(beta_min, beta_max, n_beta, endpoint=False):
 # Call the external function to get a list of energies
            model_energies = cetri_centri(np.array([alpha, beta, lauks]), tikaienergijas=True)

 # Merge close peaks
            merged_peaks = merge_close_peaks(model_energies)

 # Count the number of unique peaks
            gar = len(merged_peaks)

# Convert alpha and beta to radians for spherical coordinates
            coords = (np.deg2rad(alpha), np.deg2rad(beta))

             # --- Area Calculation ---
            # The area of a small patch on a sphere is dA = r² * cos(alpha) * d_alpha * d_beta.
            d_alpha = np.deg2rad(ik_pec_lenkis)
            d_beta = np.deg2rad((beta_max - beta_min) / n_beta)
            area_per_point = lauks**2 * np.cos(np.deg2rad(alpha)) * d_alpha * d_beta
            
            # Categorize the coordinates and calculate the total and specific areas
            if gar == 24:
                vert["24"].append(coords)
                total_area_24_only += area_per_point
            elif gar == 23:
                vert["23"].append(coords)
            elif 21 <= gar <= 22:
                vert["22-21"].append(coords)
            elif 15 <= gar <= 20:
                vert["20-15"].append(coords)
            elif 10 <= gar <= 15:
                vert["15-10"].append(coords)
            else:
                vert["under 10"].append(coords)

            total_area += area_per_point

    area_results.append((lauks, total_area))
    area_24_only_results.append((lauks, total_area_24_only))

# --- Plotting (for the last lauks value only) ---
if lauks == lauks_values[-1]:
    # 3D Plotting
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    for label, coords in vert.items():
        if len(coords) == 0:
            continue
        coords = np.array(coords)
        alphas, betas = coords[:, 0], coords[:, 1]
        x = np.cos(alphas) * np.cos(betas) * lauks
        y = np.cos(alphas) * np.sin(betas) * lauks
        z = np.sin(alphas) * lauks
        ax.scatter(x, y, z, s=20, label=label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(title="Peak count bins", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title("Peak count groups on sphere")
    plt.tight_layout()
    plt.show()

    # 2D Plotting (alpha vs beta in degrees)
    plt.figure(figsize=(9, 6))
    for label, coords in vert.items():
        if len(coords) == 0:
            continue
        coords = np.array(coords)
        alphas_deg = np.rad2deg(coords[:, 0])
        betas_deg = np.rad2deg(coords[:, 1])
        plt.scatter(alphas_deg, betas_deg, label=label, s=10)
    plt.xlabel('Alpha Angle (degrees)')
    plt.ylabel('Beta Angle (degrees)')
    plt.title('Peak count groups in 2D projection')
    plt.legend(title="Peak count bins")
    plt.grid(True)
    plt.show()

# --- Plotting final results ---
lauks_values_final = [item[0] for item in area_24_only_results]
area_24_only_results_final = [item[1] for item in area_24_only_results]
plt.figure(figsize=(9, 6))
plt.bar(lauks_values_final, area_24_only_results_final)
plt.xlabel('Lauks Value (Sphere Radius)')
plt.ylabel('Area for 24 Peaks')
plt.title('Area of 24-Peak regions for different Lauks values')
plt.xticks(lauks_values_final)
plt.grid(axis='y')
plt.show()