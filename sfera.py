import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main_func import cetri_centri, sign_dati

lauks = 46
ik_pec_lenkis = 1
tol = 1

# Angle range limits
alpha_min = -45
alpha_max = 0
beta_min = 0
beta_max = 45

vert = {
    "24":[],
    "23":[],
    "22-21":[],
    "20-15":[],
    "15-10":[],
    "under 10":[]
} 


def merge_close_peaks(energies):
    """Merge peaks that are within `tol` of each other."""
    if len(energies) == 0:
        return []
    energies = np.sort(energies)   # sort ascending
    merged = [energies[0]]
    for e in energies[1:]:
        if abs(e - merged[-1]) <= tol:
            # merge into average
            merged[-1] = (merged[-1] + e) / 2
        else:
            merged.append(e)
    return merged


idx = 0
for alpha in range(alpha_min, alpha_max, ik_pec_lenkis):
    # Calculate the number of beta points needed for this latitude band.
    # We use cosine to make the number of points proportional to the circumference.
    # The number of beta points approaches 0 at the poles (alpha = +/-90).
    n_beta = max(1, int((beta_max - beta_min) / ik_pec_lenkis * np.cos(np.deg2rad(alpha))))
    
    # Create the beta points for this specific latitude band.
    for beta in np.linspace(beta_min, beta_max, n_beta, endpoint=False):
        # Call the external function to get a list of energies
        model_energies = cetri_centri(np.array([alpha, beta, lauks]), tikaienergijas=True)

        merged_peaks = merge_close_peaks(model_energies)

        gar = len(merged_peaks)

        coords = (alpha * np.pi/180.0, beta * np.pi/180.0)
        if gar == 24:
            vert["24"].append(coords)
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


# fig = plt.figure(figsize=(9, 9))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_box_aspect([1, 1, 1])
i =1
grad = []
for label, coords in vert.items():
    if len(coords) == 0:
        continue
    coords = np.array(coords)
    alphas, betas = coords[:, 0], coords[:, 1]

    alphagrad = alphas  * 180.0/np.pi
    betagrad = betas *180.0/np.pi
    grad.append((alphagrad,betagrad))
    # spherical → Cartesian (radius = 30 for visibility)
    x = np.cos(alphas) * np.cos(betas) * lauks
    y = np.cos(alphas) * np.sin(betas) * lauks
    z = np.sin(alphas) * lauks

    #ax.scatter(x, y, z, s=20, label=label)
    i+=1
    if i==2:
        break

# ax.legend(title="Peak count bins", loc="upper left", bbox_to_anchor=(1.02, 1.0))
# ax.set_title("Peak count groups on sphere")

# plt.tight_layout()
# plt.show()

for i in grad:
    plt.scatter(i[0],i[1])
plt.scatter(-25,22.5, color = "red")
plt.show()

