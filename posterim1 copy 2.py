import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
mp.mp.dps = 60
from main_func import cetri_centri, ipasvertibas, Plane_stress_tensor, Aproximated_stress,dT_to_D,Stress_tensor

plt.figure(figsize=(7,4))

bval = np.linspace(0.001,1, 50)
D= 2870.0
Temp = [0]
col = ["red", "green"]
aa = []
bb = []
cc=[]
alfa = np.deg2rad(0)
theta = np.deg2rad(0)

for dT in Temp:
    valsa = []
    valb = []
    for b in bval:
        bx=b*np.sin(alfa)+b*np.cos(theta)
        by=b*np.cos(alfa)+b*np.cos(theta)
        bz=b*np.cos(alfa)
        mxa,mya,mza = Aproximated_stress(dT)
        mxb,myb, mzb, nxb,nyb = Stress_tensor(0,0.0,0.0,0.05,-0.05,-0.05)
        lvlsa = ipasvertibas(bx,by,bz,D+dT_to_D(dT),mxa,mya,mza,nuclear=False)
        lvlsa = ipasvertibas(bx,by,bz,D,mxb,myb, mzb,Nx=nxb,Ny=nyb,nuclear=False) # Nx=nxb,Ny=nyb
        # valb.append(lvlsb)
        valsa.append(lvlsa)
    valsa = np.array(valsa, dtype=object)  # nevis float64
    valb  = np.array(valb,  dtype=object)
    pirm = valsa[:, 1] - valsa[:, 0]
    otr  = valsa[:, 2] - valsa[:, 0]
    bb.append(pirm)
    bb.append(otr)
    aa.append(valsa[:, 2] - valsa[:, 1])
    # cc.append(valb[:, 2] - valb[:, 1])
# aa.append(aa[0]-aa[1])
# cc.append(cc[0]-cc[1])
plt.plot(bval,pirm, label=r'$\sigma = f(\Delta T) and B = f(E_{+1} - E_{-1})$' ,linewidth = 5,color = "red")
plt.plot(bval,otr, label=r'$\sigma = f(\Delta T) and B = f(E_{+1} - E_{-1})$' ,linewidth = 5,color = "red")
# plt.plot(bval, cc[1], label =r'$B = f(E_{+1} - E_{-1})$',linewidth = 5,color = "green")
# plt.plot(bval, cc[0], label =r'$B = f(E_{+1} - E_{-1})$',linewidth = 5,color = "green")
# plt.plot(bval,-bb[1]+bb[3], label ='previous aproach',linewidth = 5, color = "blue")
plt.yscale("log")
plt.ylabel("Energy, MHz")
plt.xlabel("B,G")
plt.grid()
plt.title("1K temperature difference in different approaches",fontweight='bold')
plt.legend()
plt.tight_layout()

# ratio_cc = (-bb[1] + bb[3]) / cc[2]
# ratio_aa = (-bb[1] + bb[3]) / aa[2]

# print(f"{float(pirm[-1]-otr[-1]):.6e}",f"{float(ratio_cc[-1]):.6e}", f"{float(ratio_aa[-1]):.6e}")

plt.show()