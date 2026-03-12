import numpy as np
Sx = np.array(((0.,1,0),
              (1,0,1),
              (0,1,0)))/np.sqrt(2)

Sy = 1j*np.array(((0.+0j,-1,0),
              (1,0,-1),
              (0,1,0)))/np.sqrt(2)

Sz = np.array(((1,0.0,0),
              (0,0,0),
              (0,0,-1)))

agper = -2.7 #Mhz 14N
agparal = -2.14

g_bora = 28.03*0.1

Sx2 = Sx @ Sx
Sy2 = Sy @ Sy
Sz2 = Sz @ Sz
print(Sz2+Sy2-Sx2)