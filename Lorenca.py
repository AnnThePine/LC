import numpy as np
import matplotlib.pyplot as plt

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
bora = 13.996


def ipasvertibas(Bx, By, Bz, D=2.88):
    Dzz = D * 2 / 3
    Dxx = Dyy = -Dzz / 2  # From Dxx + Dyy + Dzz = 0
    SDS = Dxx * (Sx @ Sx) + Dyy * (Sy @ Sy) + Dzz * (Sz @ Sz)
    Hamitonis = SDS + (g * bora *(Bz * Sz + Bx*Sx + By*Sy))

    eigen = np.linalg.eigh(Hamitonis)
    eigenvalues = eigen[0]
    eigenvectors = eigen[1]

    # print(eigenvalues)
    # print(eigenvectors)
    
    return eigenvalues, eigenvectors

def viens_centrs(Virziens, mag):

    lauks = np.array([mag * Virziens[0], mag * Virziens[1], mag * Virziens[2]])

    #print(lauks[:, 0])
    en, vectors = ipasvertibas(*lauks)

    energijas = np.sort(en)

    # print(energijas.shape)
    energiju_limeni = np.array((energijas[1]-energijas[0],energijas[2]-energijas[0]))
    
    return energiju_limeni
        #return energijas

def Spektrs(energijas, platums=1):

    if type(energijas) == list:
        energijas = np.concatenate(energijas)


    # You can choose to use energijas range or fixed sak/beigu
    mini = min(energijas) - 10 
    maxi = max(energijas) + 10

    freq_range = np.linspace(mini, maxi, 5000)  # GHz
    odmr_signal = np.zeros_like(freq_range)

    for freq in energijas:
        lorentz = (platums / (2 * np.pi)) / ((freq_range - freq)**2 + (0.5 * platums)**2)
        odmr_signal += lorentz

    # Optionally invert for dip-style ODMR
    odmr_signal = 1 - odmr_signal / np.max(odmr_signal)

    return freq_range,odmr_signal


def plot(z): 
    x,y = z 
    plt.plot(x,y)
    plt.show()

def grad_vect(alfa, beta):
    alfa = np.deg2rad(alfa)
    beta   = np.deg2rad(beta)

    x = np.sin(alfa) * np.cos(beta)
    y = np.sin(alfa) * np.sin(beta)
    z = np.cos(alfa)

    v = np.array([x, y, z], dtype=float)

    return v


def asis(z_axis):
    """Izveido lokālo (x,y,z), kur z = NV virziens."""
    z_axis = z_axis / np.linalg.norm(z_axis)  # NV normalizēts
    # Izvēlamies pagaidu vektoru, kas nav paralēls z
    tmp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(tmp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack([x_axis, y_axis, z_axis])

def cetri_centri(lauka_virziens, mag):

    #lauka virziens ir mag lauka virziena matrica

    c = 1/np.sqrt(3.0)  # 1/√3

    R = np.array([
        [(1 + c)/2, (c - 1)/2, -c],
        [(c - 1)/2, (1 + c)/2, -c],
        [c,         c,         c ]
    ], dtype=np.float64)

    #definējam nv asu virzienu 

    NV1z = R@np.array(((1),(1),(1)))/np.sqrt(3)
    NV1x = R@np.array(((1),(1),(1)))/np.sqrt(3)
    NV1y = R@np.array(((1),(1),(1)))/np.sqrt(3)
    
    NV2 = R@np.array(((-1),(1),(-1)))/np.sqrt(3)
    NV3 = R@np.array(((1),(-1),(-1)))/np.sqrt(3)
    NV4 = R@np.array(((-1),(-1),(1)))/np.sqrt(3)

    NV_centri = [
        R @ np.array([ 1,  1,  1]) / np.sqrt(3),
        R @ np.array([-1,  1, -1]) / np.sqrt(3),
        R @ np.array([ 1, -1, -1]) / np.sqrt(3),
        R @ np.array([-1, -1,  1]) / np.sqrt(3),
    ]

    rezultati = []
    for NV in NV_centri:
        R_local = asis(NV)     # 3x3
        B_local = R_local @ lauka_virziens      # (Bx, By, Bz)
        en = viens_centrs(B_local, mag)
        rezultati.append((en))


    return rezultati



plot(Spektrs(cetri_centri(grad_vect(91,70),0.3)))