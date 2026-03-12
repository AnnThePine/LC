import matplotlib.pyplot as plt
import numpy as np
from main_func import cetri_centri, ipasvertibas, Plane_stress_tensor, Aproximated_stress,dT_to_D,Stress_tensor

plt.figure(figsize=(7,4))

bval = np.linspace(0.001,10, 30)
D= 2850.0
Temp = [0,1]
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
        mxb,myb, mzb, nxb,nyb = Stress_tensor(1,1,1,1,1,1)
        lvlsa = ipasvertibas(bx,by,bz,D+dT_to_D(dT),mxa,mya,mza,nuclear=False)
        lvlsb = ipasvertibas(bx,by,bz,D+dT_to_D(dT),mxb,myb, mzb,nuclear=False) # Nx=nxb,Ny=nyb
        valb.append(lvlsb)
        valsa.append(lvlsa)
    valsa = np.array(valsa, dtype=np.float64)
    valb = np.array(valb, dtype=np.float64)
    pirm = valsa[:, 1] - valsa[:, 0]
    otr  = valsa[:, 2] - valsa[:, 0]
    bb.append(pirm)
    bb.append(otr)
    aa.append(valsa[:, 2] - valsa[:, 1])
    cc.append(valb[:, 2] - valb[:, 1])
aa.append(aa[0]-aa[1])
cc.append(cc[0]-cc[1])
plt.plot(bval, aa[2], label = "reference temp")
plt.plot(bval, cc[2], label = "without temperature strain relation")
plt.plot(bval,-bb[0]+bb[2])
plt.plot(bval,-bb[1]+bb[3])
plt.yscale("log")
plt.ylabel("Energy, GHz")
plt.xlabel("B,T")
plt.grid()
plt.title("100K temperature difference in transition energy difference",fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()