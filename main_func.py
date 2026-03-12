import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy.signal import find_peaks

amplituda = 20

platums = 1.5

punkti = 5000

mini = 2650
maxi = 3100

# nuclear spin I = 1 operators (3x3)
Ix = np.array(((0., 1, 0),
               (1, 0, 1),
               (0, 1, 0)))/np.sqrt(2)

Iy = 1j*np.array(((0., -1, 0),
                  (1,  0, -1),
                  (0,  1,  0)))/np.sqrt(2)

Iz = np.array(((1, 0., 0),
               (0, 0., 0),
               (0, 0., -1)))

D = 2850.0  # MHz

Q = -4.96       # MHz
gamma_N = 0.3077e-3  # MHz/G   (0.3077 kHz/G)


# agparal = -1.7 #MHz būs param , maina videjo piki
# agper = -7 #MHz būs param, maina attālumu starp mazajām energijām


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

alfa_const_dim = 1.06*10e-6#1/K

C = 1076+125-((2*125^2)/1076)

Sx2 = Sx @ Sx
Sy2 = Sy @ Sy
Sz2 = Sz @ Sz

Sxy = Sx@Sy
Syx = Sy@Sx

Sxz = Sx@Sz
Szx = Sz@Sx

Syz = Sy@Sz
Szy = Sz@Sy

Mxx = Sx2-Sy2
Myy = Sxy+Syx

Nxx= Sxz+Szx
Nyy = Syz+Szy

I_n = np.eye(3)

SIz = np.kron(Sz, Iz)
SI = np.kron(Sx, Ix) + np.kron(Sy, Iy)

# NV centru virzieni 54.7 grādu leņķī
nvcentri = np.array([[-1, -1, 1],
                        [1, 1, 1],
                        [-1, 1, -1],
                        [1, -1, -1]]) / np.sqrt(3)

# Aprēķina divus vienības vektorus, kas ir perpendikulāri nvcentriem
nvcentrix = np.array([np.cross(nvcentri[0, :], nvcentri[1, :]),
                        np.cross(nvcentri[0, :], nvcentri[1, :]),
                        np.cross(nvcentri[2, :], nvcentri[3, :]),
                        np.cross(nvcentri[2, :], nvcentri[3, :])])

for i in range(4):
    nvcentrix[i, :] /= np.linalg.norm(nvcentrix[i, :])

nvcentriy = np.array([np.cross(nvcentri[0, :], nvcentrix[0, :]),
                        np.cross(nvcentri[1, :], nvcentrix[1, :]),
                        np.cross(nvcentri[2, :], nvcentrix[2, :]),
                        np.cross(nvcentri[3, :], nvcentrix[3, :])])

def ipasvertibas(Bx, By, Bz,D, Mx, My, Mz, Nx = 0, Ny = 0 , nuclear = True):
    # nuclear identity (3x3)

    # electron zero-field splitting (MHz)
    Dzz = D * 2 / 3
    Dxx = Dyy = -Dzz / 2
    SDS = Dxx * (Sx2) + Dyy * (Sy2) + Dzz * (Sz2)

    # electron Zeeman (MHz)
    Zeeman  = g_bora * (Bz*Sz + Bx*Sx + By*Sy)

    # strain term (MHz)  --- new
    Strain = Mz*(Sz2)+Mx*Mxx+My*Myy+Nx*Nxx+Ny*Nyy

    # full electron part in full space (3 electron ⊗ 3 nuclear = 9x9)
    H_elec_full = SDS + Zeeman + Strain
    if nuclear:
    # nuclear-only part (MHz)
        H_nuc = - gamma_N * (Bx*Ix + By*Iy + Bz*Iz)

        H_nuc_full = np.kron(I_n, H_nuc)

    # hyperfine interaction (MHz)
        H_hyperfine = (agparal * SIz + agper   * SI)

        Hamiltonian = np.kron(H_elec_full, I_n) + H_nuc_full + H_hyperfine
    else:
        Hamiltonian = H_elec_full
    #print(Hamiltonian)
    eigenvalues = np.linalg.eigvalsh(Hamiltonian)
    return np.sort(eigenvalues)

def Aproximated_stress(dT,b=7.1,a1=-11.7):
    Mx = -b*alfa_const_dim*C
    Mz = -a1*alfa_const_dim*C
    My = 0
    #Sz2+Sy2-Sx2
    return Mx, My, Mz

def dT_to_D(dT):
    return 0.075*dT


def Plane_stress_tensor(sxx, sxy, syy, a1=-11.7,a2=6.5,b=7.1,c=-5.4): 
    #Mhz/GPa(Barfuss), assumes plane stress, Nx, Ny not defined, needs further studies
    Mx = -b*(sxx +syy) + c*2*sxy
    My = np.sqrt(3)*b*(sxx - syy)
    Mz = a1*(sxx + syy) + 2*a2* sxy
    return Mx, My, Mz

def Stress_tensor(sxx,sxy,syy,szz,sxz,syz,a1=-11.7,a2=6.5,b=7.1,c=-5.4, d = 2, e = 4):
    Mx = b * (2*szz - sxx - syy) + c * (2*sxy - syz - sxz)
    My = np.sqrt(3) * b * (sxx - syy) + np.sqrt(3) * c * (syz - sxz)
    Mz = a1 * (sxx + syy + szz) + 2 * a2 * (syz + sxz + sxy)
    Nx = d * (2*szz - sxx - syy) + e * (2*sxy - syz - sxz)
    Ny = np.sqrt(3) * d * (sxx - syy) + np.sqrt(3) * e * (syz - sxz)
    return Mx, My, Mz, Nx, Ny


def cetri_centri(sferiskas_koord, D, dirrr = [1,0], P=0.03,a1 = 5,a2= -3,b = -2,c =7, vajagrange = True, griezums="100", tikaienergijas = False):

    # Ja lauka_kompoentes ir lmfit Parameters objekts, izvelkam vērtības
    if isinstance(sferiskas_koord, lmfit.parameter.Parameters):
        sferiskas_koord = np.array([
            sferiskas_koord['Bvert'].value,
            sferiskas_koord['Bhor'].value,
            sferiskas_koord['Babs'].value])
        
    
    sferiskas_koord = np.array(sferiskas_koord)

    # drošības pārbaude, lai redzētu, ja saņem nepareizu formu
    if sferiskas_koord.size < 3:
        raise ValueError(f"sferiskas_koord jāsatur 3 vērtības (vert, hor, abs). Saņemts: {sferiskas_koord}")


    lauka_kompoentes = grad_vect(sferiskas_koord[0], sferiskas_koord[1])
    lauka_kompoentes = lauka_kompoentes*sferiskas_koord[2]
    all_energijas = []

    if griezums == "100":
        
        stress_dir = grad_vect(*dirrr)
        # Aprēķina enerģijas katram centram un apkopo kopā
        for NV in range(4):

            R_k = np.vstack([nvcentrix[NV], nvcentriy[NV], nvcentri[NV]])

            sigma = P * np.outer(stress_dir, stress_dir)

            sigma_nv = R_k @ sigma @ R_k.T

            #####Mx, My, Mz = Stress_tensor(sigma_nv, a1,a2,b,c)

            nvkomp = np.array([
                np.dot(nvcentrix[NV, :], lauka_kompoentes),
                np.dot(nvcentriy[NV, :], lauka_kompoentes),
                np.dot(nvcentri[NV, :], lauka_kompoentes)])
            #en = ipasvertibas(*nvkomp,Mx, My, Mz, D)
            #energijas = en[1:] - en[0]
            #all_energijas.append(energijas)

        

    all_energijas = np.concatenate(all_energijas)

    #strain virzieni

    if tikaienergijas:
        ener = [e for e in all_energijas if mini <= e <= maxi]
        #ener = all_energijas

        if len(ener) == 0:
            print("kaut kas nogāja galīgi garām")
            return []

        # Sakārto pēc lieluma, lai vieglāk apvienot
        ener.sort()

        merged_peaks = []
        current_group = [ener[0]]

        for ee in ener[1:]:
            if abs(ee - current_group[-1]) <= 3.0:
                current_group.append(ee)
            else:
                avg = sum(current_group) / len(current_group)
                merged_peaks.append(avg)
                current_group = [ee]

        # pievieno pēdējo grupu
        if current_group:
            avg = sum(current_group) / len(current_group)
            merged_peaks.append(avg)

        return merged_peaks

    else:

        #maxi = np.max(all_energijas) + 20

        freq_range = np.linspace(mini, maxi, punkti)
        odmr_signal = np.zeros_like(freq_range)

        for freq in all_energijas:
            gamma = platums / 2
            lorentz = -gamma**2 / ((freq_range - freq)**2 + gamma**2)
            odmr_signal += lorentz

        odmr_signal /= np.min(odmr_signal)

        if vajagrange:
            return freq_range,odmr_signal
        else:
            return odmr_signal


def grad_vect(alfa, beta):
    alfa = np.deg2rad(alfa)
    beta   = np.deg2rad(beta)

    x = np.sin(alfa) * np.cos(beta)
    y = np.sin(alfa) * np.sin(beta)
    z = np.cos(alfa)

    v = np.array([x, y, z], dtype=float)

    return v


def vect_grad(v):
    x, y, z = v

    print(x,y,z)
    # compute length (magnitude)
    r = np.sqrt(x**2 + y**2 + z**2)

    alfa = np.arccos(z/r)             # radians
    beta = np.arctan2(y, x)       # radians

    # convert to degrees
    alfa = np.rad2deg(alfa)
    beta = np.rad2deg(beta)

    return [alfa, beta, r]


def plot(x,y):
    plt.plot(x,y)
    plt.show()


def sign_dati(freq_range, signal, prominence=0.1, distance=3, width=0.8):
    idx, props = find_peaks(signal,
                            prominence=prominence,
                            distance=distance,
                            width=width)
    peak_frequencies = freq_range[idx]
    peak_intensities = signal[idx]

    # Filtrs: tikai pīķi, kas lielāki par 1/5 no lielākā pīķa
    mask = peak_intensities > peak_intensities.max() / 5

    frequencies = peak_frequencies[mask]
    intensities = peak_intensities[mask]

    return frequencies.tolist(), intensities.tolist()

def Vid_kvadr(x,y,square = True):
    if len(x) == len(y):
        if square:
            x=np.sort(x)
            y=np.sort(y)
            return (x-y)**2
        else:
            x=np.sort(x)
            y=np.sort(y)
            return (x-y)**6
    else:
        return np.inf