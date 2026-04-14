import matplotlib.pyplot as plt
import numpy as np
from main_func import Temperature_dependance, ipasvertibas

Tvals = np.linspace(-10, 100, 60)

b = 0.3
alfa = 0
theta = 0
val1 = []
val2 = []
starp = []

for i in Tvals:
    d,q,apar,aper = Temperature_dependance(i)

    bx=b*np.sin(alfa)+b*np.cos(theta)
    by=b*np.cos(alfa)+b*np.cos(theta)
    bz=b*np.cos(alfa)

    eigval = ipasvertibas(bx,by,bz,d,0,0,0,agper = aper,agparal=apar,Q=q,nuclear=False)
    print(eigval)
    val1.append(eigval[1]-eigval[0])
    val2.append(eigval[2]-eigval[0])
    starp.append(-eigval[1]+eigval[2])

plt.plot(Tvals,val1, color = "purple",linewidth = 5,label= "|0⟩ ⟶|-1⟩")
plt.plot(Tvals,val2, color = "purple",linewidth = 5,label= "|0⟩ ⟶|+1⟩")
#plt.plot(Tvals,starp,color = "red", label= "|+1⟩ ⟶|-1⟩",linewidth = 5)
plt.xlabel("dT, K (T0 = 297K)",fontsize=14)
plt.ylabel("E, MHz",fontsize=14)
plt.grid()
plt.title("Temperature induced error propogation within energy levels",fontweight='bold',fontsize=20)
plt.legend(fontsize=14)
plt.show()
