import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from main_func import cetri_centri, ipasvertibas

plt.figure(figsize=(7,4))

bval = np.linspace(0,0.1, 30)
D= [2850.0,2850.0+7.5]
num = [0,1]
col = ["red", "green"]
aa = []
bb = []
for a in num:
    vals = []
    for b in bval:
        lvls = ipasvertibas(0,0,b,-0.0, 0, 0, D[a],nuclear=False)
        vals.append(lvls)
    vals = np.array(vals, dtype=np.float64)
    pirm = vals[:, 1] - vals[:, 0]
    otr  = vals[:, 2] - vals[:, 0]
    bb.append(pirm)
    bb.append(otr)
    aa.append(vals[:, 2] - vals[:, 1])
    print(aa)
aa.append(aa[0]-aa[1])
print("min/max gap21:", vals.min(), vals.max())
print(aa[2])
plt.plot(bval, aa[2],linewidth = 5, label = "reference temp")
plt.plot(bval,-bb[0]+bb[2])
plt.plot(bval,-bb[1]+bb[3])

plt.ylabel("Energy, GHz")
plt.ticklabel_format(useOffset=False)
plt.xlabel("B,T")
plt.grid()
#plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.title("100K temperature difference in transition energy difference",fontweight='bold')
# plt.legend()
plt.tight_layout()
plt.show()