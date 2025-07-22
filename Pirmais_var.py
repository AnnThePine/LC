import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment


Sx = np.array(((0.,1,0),
              (1,0,1),
              (0,1,0)))/np.sqrt(2)

Sy = 1j*np.array(((0.+0j,-1,0),
              (1,0,-1),
              (0,1,0)))/np.sqrt(2)

Sz = np.array(((1,0.0,0),
              (0,0,0),
              (0,0,-1)))


g = 2.0028
bora = 14


def ipasvertibas(Bx, By, Bz, D=2.88):

    Dzz = D * 2 / 3
    Dxx = Dyy = -Dzz / 2  # From Dxx + Dyy + Dzz = 0

    SDS = Dxx * (Sx @ Sx) + Dyy * (Sy @ Sy) + Dzz * (Sz @ Sz)

    Hamitonis = SDS + (g * bora *(Bz * Sz + Bx*Sx + By*Sy))

    eigen = np.linalg.eigh(Hamitonis)
    #print (eigen)
    eigenvalues = eigen[0].real
    #eigenvalues = np.sort(eigenvalues)
    #print(eigenvalues)
    

    return eigenvalues[0], eigenvalues[1], eigenvalues[2]

def laukanovietojums(B,alfa,beta):
    alfa_rad = np.radians(alfa)
    beta_rad = np.radians(beta)

    x = B * np.sin(alfa_rad) * np.cos(beta_rad)
    y = B * np.sin(alfa_rad) * np.sin(beta_rad)
    z = B * np.cos(alfa_rad)

    return x, y, z

def enstarp(B, alfa, beta):
     x,y,z = laukanovietojums(B,alfa,beta)
     i0,i1,i2 = ipasvertibas(x,y,z)

     return i0,i1,i2
#     #if alfa == 0: 
#     #return i0-i1, i2-i1
#     # else:
#     return i1-i0, i2-i0

#     #print(i0,i1,i2)
#     #return i1-i0, i2-i0
#     #return i0-i1, i2-i1


def energijas1(alfa,beta, plot = False):

    alfa = np.radians(alfa)
    beta = np.radians(beta)


    B = np.array(np.linspace(0,0.2,100))
    xlauks = []
    ylauks = []
    zlauks = []

    for mag in B:
        xl,yl,zl = laukanovietojums(mag, alfa, beta)
        xlauks.append(xl)
        ylauks.append(yl)
        zlauks.append(zl)


    sar0 = []
    sar1 = []
    sar2 = []

    for vert, i in enumerate(B):
        v0,v1,v2 = ipasvertibas(xlauks[vert],ylauks[vert],zlauks[vert])
        sar0.append(v0)
        sar1.append(v1)
        sar2.append(v2)
    
    starp = sar0[0]

    starp0 = sar0-starp
    starp1 = sar1-starp
    starp2 = sar2-starp

    if plot:
        plt.plot(B,starp0)
        plt.plot(B,starp1)
        plt.plot(B,starp2)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return starp0, starp1, starp2, B



def cetricentri(B,alfa,beta):
#     #pirmais vnk tāpat
    en1 = enstarp(B, alfa,beta)

#     #otrais ybeta=0,x alfa-109.5
    en2 = enstarp(B, alfa-109.5, beta)

#     # treshais, ceturtais alfa-109.5, beta _+120
#     #treshais
    en3 = enstarp(B,alfa-109.5, beta+120)

#     #ceturtais
    en4 = enstarp(B,alfa-109.5, beta-120)

    energijas = [en1,en2,en3,en4]

    return energijas

def sakarto(a,b,no,lidz):
    
    lauks = np.linspace(no,lidz, 500)

    energijas = []
    for B in lauks:
        centri = cetricentri(B, a, b)
        energijas.append(centri)

    print(energijas)
    # 2. Inicializē izsekošanas struktūru
    energijas_traces = [[] for _ in range(8)]  # 8 līmeņi

    # 3. Pirmais solis – pieņem sākuma secību
    pirmas = sum(energijas[0], ())  # flatten [[a,b,c],[d,e,f]...] -> (a,b,...,h)
    for i in range(8):
        energijas_traces[i].append(pirmas[i])

    prev = np.array(pirmas)

    # 4. Pārējie soļi – sakārto pēc tuvuma iepriekšējam solim
    for i in range(1, len(lauks)):
        current = sum(energijas[i], ())  # flatten uz (8,) masīvu
        current = np.array(current)

        # Attālumu matrica
        distances = np.abs(prev[:, None] - current[None, :])
        row_ind, col_ind = linear_sum_assignment(distances)
        sorted_current = current[col_ind]

        for j in range(8):
            energijas_traces[j].append(sorted_current[j])

        prev = sorted_current

    normalizetie_traces = []
    
    grupas = [(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 6)]
    
    for ground_idx, excited_idx in grupas:
        trace_ground = np.array(energijas_traces[ground_idx])
        trace_excited = np.array(energijas_traces[excited_idx])
        normalizeta = trace_excited - trace_ground
        normalizetie_traces.append(normalizeta)  

    for i, trace in enumerate(normalizetie_traces):
            plt.plot(lauks, trace)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # for b in energijas:
    #     for centri in b:
    #         for centrs in centri: 
    #             for en in centrs:
                    
                

sakarto(0,0,0,0.3)



def vairakimainigalauka(a,b,no,lidz):
    energijas = [[],[],[],[],[],[],[],[]]
    
    lauks = np.linspace(no,lidz, 500)

    for i in lauks:
        en = cetricentri(i, a, b)
        for c in range(8):
            energijas[c].append(en[c])


    for c, lim in enumerate(energijas):
        plt.plot(lauks,energijas[c])

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
vairakimainigalauka(0,0,0,0.3)


def Spektrs(energijass,sak =100,  beigu=1000, platums=3 ):
    freq_range = np.linspace(sak, beigu, 5000)  # GHz
    odmr_signal = np.zeros_like(freq_range)

    for i,freq in enumerate(energijass):
        a = -0.5*platums/(np.pi*((freq_range-freq)**2+(0.5*platums)**2))
        odmr_signal += a
        # break
    
    return odmr_signal, freq_range

# en = cetricentri(5, 50,20)
# print(en)
# odmr, freq = Spektrs(en,100,150000,0.05)

# plt.figure(figsize=(12,5))

# #plt.xscale("log")
# plt.plot(freq, odmr)
# plt.tight_layout()
# plt.show()

