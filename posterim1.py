import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from main_func import cetri_centri, ipasvertibas

plt.figure(figsize=(7,4))

bval = np.linspace(0,0.1, 30)
D= [2875.0,2875.0-0.075,2850.0,2850.0+0.075]
num = [0,1]
col = ["red", "green", "blue", "cyan"]

for a in num:
    vals = []
    for b in bval:
        lvls = ipasvertibas(0,0,b,-0.0, 0, 0, D[a],nuclear=False)
        vals.append(lvls)
    vals = np.array(vals)
    pirm = vals[:,1]-vals[:,0]
    otr = vals[:,2]-vals[:,0]
    plt.plot(bval,pirm/1000,color = col[a], linewidth = 5)
    plt.plot(bval,otr/1000,color = col[a], linewidth = 5)
plt.ylabel("Energy, GHz")
plt.ticklabel_format(useOffset=False)
plt.xlabel("B,T")
plt.grid()
#plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.title("1K temperature difference induced ZFS drift",fontweight='bold')
# plt.legend()
plt.tight_layout()
plt.show()