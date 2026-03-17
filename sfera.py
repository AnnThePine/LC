import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from main_func import cetri_centri, sign_dati, grad_vect, vect_grad

lauks = 15
ik_pec_lenkis = 1

# Angle range limits
alpha_min = 0
alpha_max = 180
beta_min = 0
beta_max = 360

vert = {
    "24":[],
    "23":[],
    "22-21":[],
    "20-15":[],
    "15-11":[],
    "10":[],
    "under 10":[]
} 

idx = 0
for alpha in range(alpha_min, alpha_max, ik_pec_lenkis):
    for beta in range(beta_min, beta_max, ik_pec_lenkis):
        
        freq, odmr = cetri_centri(np.array([alpha, beta, lauks]))
        peaks, peaky = sign_dati(freq, odmr)

        gar = len(peaks)

        coords = (alpha, beta) #grad
        if gar == 24:
            vert["24"].append(coords)
        elif gar == 23:
            vert["23"].append(coords)
        elif 21 <= gar <= 22:
            vert["22-21"].append(coords)
        elif 15 <= gar <= 20:
            vert["20-15"].append(coords)
        elif 11 <= gar <= 15:
            vert["15-11"].append(coords)
        elif gar==10:
            vert["10"].append(coords)
        else:
            vert["under 10"].append(coords)


fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
i =1
grad = []
for label, coords in vert.items():
    if len(coords) == 0:
        continue
    coords = np.array(coords)
    alphas, betas = coords[:, 0], coords[:, 1]
    vect =grad_vect(alphas,betas)
    # alphagrad = alphas  * 180.0/np.pi
    # betagrad = betas *180.0/np.pi
    grad.append((alphas,betas, label))
    # # spherical → Cartesian (radius = 30 for visibility)
    # x = lauks * np.cos(alphagrad) * np.cos(betagrad)
    # y = lauks * np.cos(alphagrad) * np.sin(betagrad)
    # z = lauks * np.sin(alphagrad)

    ax.scatter(*vect, s=20, label=label)
    i+=1




ax.legend(title="Peak count bins", loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.set_title("Peak count groups on sphere")

plt.tight_layout()
plt.show()

for i in grad:
    print(i)
    plt.scatter(i[0],i[1], label = f"Number of peaks:{i[2]}, amount: {len(i[0])}")

plt.legend()
plt.show()

