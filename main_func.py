import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy.signal import find_peaks

amplituda = 20

platums = 2

punkti = 5000

mini = 2650
maxi = 3000

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

D = 2870.0  # MHz

Q = -4.96       # MHz
gamma_N = 0.3077e-3  # MHz/G   (0.3077 kHz/G)


agparal = -1.7 #MHz būs param , maina videjo piki
agper = -7 #MHz būs param, maina attālumu starp mazajām energijām


Sx = np.array(((0.,1,0),
              (1,0,1),
              (0,1,0)))/np.sqrt(2)

Sy = 1j*np.array(((0.+0j,-1,0),
              (1,0,-1),
              (0,1,0)))/np.sqrt(2)

Sz = np.array(((1,0.0,0),
              (0,0,0),
              (0,0,-1)))

g_bora = 28.03*0.1


def ipasvertibas(Bx, By, Bz, Nx=2.0):
    I_n = np.eye(3)  # nuclear identity (3x3)
    
    # electron zero-field splitting (MHz)
    Dzz = D * 2 / 3
    Dxx = Dyy = -Dzz / 2
    SDS = Dxx * (Sx @ Sx) + Dyy * (Sy @ Sy) + Dzz * (Sz @ Sz)

    # electron Zeeman (MHz)
    Zeeman = g_bora * (Bz*Sz + Bx*Sx + By*Sy)

    # strain term (MHz)  --- new
    Strain = Nx * (Sx @ Sz + Sz @ Sx)

    # full electron part in full space (3 electron ⊗ 3 nuclear = 9x9)
    H_elec_full = np.kron(SDS + Zeeman + Strain, I_n)

    # nuclear-only part (MHz)
    H_nuc = Q * (Iz @ Iz) - gamma_N * (Bx*Ix + By*Iy + Bz*Iz)
    H_nuc_full = np.kron(np.eye(3), H_nuc)

    # hyperfine interaction (MHz)
    H_hyperfine = (
        agparal * np.kron(Sz, Iz) +
        agper   * (np.kron(Sx, Ix) + np.kron(Sy, Iy))
    )

    Hamiltonian = H_elec_full + H_nuc_full + H_hyperfine
    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)
    return eigenvalues, eigenvectors


def cetri_centri(lauka_kompoentes, vajagrange = True, griezums="100", tikaienergijas = False):

    # Ja lauka_kompoentes ir lmfit Parameters objekts, izvelkam vērtības
    if isinstance(lauka_kompoentes, lmfit.parameter.Parameters):
        lauka_kompoentes = np.array([
            lauka_kompoentes['Bx'].value,
            lauka_kompoentes['By'].value,
            lauka_kompoentes['Bz'].value])

    lauka_kompoentes = np.array(lauka_kompoentes)

    all_energijas = []

    if griezums == "100":
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

        # Aprēķina enerģijas katram centram un apkopo kopā
        for NV in range(4):
            nvkomp = np.array([
                np.dot(nvcentrix[NV, :], lauka_kompoentes),
                np.dot(nvcentriy[NV, :], lauka_kompoentes),
                np.dot(nvcentri[NV, :], lauka_kompoentes)])
            en, _ = ipasvertibas(*nvkomp)
            energijas = en[1:] - en[0]
            all_energijas.append(energijas)

    all_energijas = np.concatenate(all_energijas)

    if tikaienergijas:
        ener = []
        for e in all_energijas:
            if mini<=e<=maxi:
                ener.append(e)
        return ener
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


def plot(x,y):
    plt.plot(x,y)
    plt.show()


def sign_dati(freq_range, signal, prominence=0.05, distance=3, width=1):
    idx, props = find_peaks(signal,
                            prominence=prominence,
                            distance=distance,
                            width=width)
    peak_frequencies = freq_range[idx]
    peak_intensities = signal[idx]
    return peak_frequencies, peak_intensities

