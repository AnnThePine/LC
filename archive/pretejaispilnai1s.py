import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy.signal import find_peaks
'''
Jāpievieno strain

Jāsalabo fitting, var;etu būt ka kkas no teslam palicis

Good luck buddy

'''
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
    return freq_range[idx]


def pretejais(energijas):
    minej = 100
    minejumi = np.random.uniform(-amplituda, amplituda, size=(3,minej))

    def residual_func(p):
        model_energies = cetri_centri(
            [p['Bx'], p['By'], p['Bz']],
            vajagrange=False,
            tikaienergijas=True
        )
        
        # Sort and filter the model's energies within the relevant range
        model_energies = np.array(model_energies)
        model_energies = model_energies[(model_energies >= mini) & (model_energies <= maxi)]
        model_energies.sort()
        
        # Merge energies that are within 3 MHz
        merged_peaks = []
        if len(model_energies) > 0:
            current_peak = model_energies[0]
            count = 1
            for i in range(1, len(model_energies)):
                if abs(model_energies[i] - current_peak) < 3.0:  # 3.0 MHz threshold
                    current_peak = (current_peak * count + model_energies[i]) / (count + 1)
                    count += 1
                else:
                    merged_peaks.append(current_peak)
                    current_peak = model_energies[i]
                    count = 1
            merged_peaks.append(current_peak)
        
        model_peaks = np.array(merged_peaks)
        data_peaks = np.sort(energijas)

        # The issue is here: the length of residuals must be constant.
        # We will pad the residuals array to a fixed size.
        # We can determine the max possible length based on the data_peaks.
        
        max_residuals = len(data_peaks) + 1 # +1 for the penalty term

        residuals = np.zeros(max_residuals)
        
        # Check for number of peaks and calculate penalty
        peak_count_diff = len(model_peaks) - len(data_peaks)
        residuals[0] = 100 * abs(peak_count_diff)
        
        # Pair the peaks and calculate residuals for each data peak
        for i, dp in enumerate(data_peaks):
            if len(model_peaks) > 0:
                idx = np.argmin(np.abs(model_peaks - dp))
                residuals[i+1] = (model_peaks[idx] - dp) * 1e6  # Convert to Hz
            else:
                residuals[i+1] = 1000  # Large value if no model peaks exist
        
        return residuals

    
    best_result = None
    best_resid = np.inf

    for minejums in minejumi:
        params = lmfit.Parameters()
        params.add('Bx', value=minejums[0], min=-amplituda, max=amplituda)
        params.add('By', value=minejums[1], min=-amplituda, max=amplituda)
        params.add('Bz', value=minejums[2], min=-amplituda, max=amplituda)

        result = lmfit.minimize(
            residual_func, params,
            method="least_squares",      # nevis "leastsq"
            kws={},                       # (nav obligāti)
            max_nfev=5000,
            ftol=1e-15, xtol=1e-15, gtol=1e-15,
            loss="soft_l1", f_scale=1.0   # robusts zudums, ~|resid|
        )


        resid = np.sum(result.residual**2)
        if resid < best_resid:
            best_resid = resid
            best_result = result

    best_params_list = list(best_result.params.valuesdict().values())

    return best_result, best_params_list


def testejam(B, alfa, beta):
    mag_lauks = grad_vect(alfa, beta)*B # mT

    freq_range, spektrs = cetri_centri(mag_lauks, True)

    energijas = sign_dati(freq_range, spektrs)


    pret, pirma_rez = pretejais(energijas)

    print(mag_lauks)
    print(pirma_rez)


    plt.plot(freq_range, spektrs, label="simulated")
    simfreq,simspektrs = cetri_centri(pirma_rez, True)
    plt.plot(simfreq, simspektrs, label="model guess", alpha = 0.5)
    peaks = sign_dati(freq_range, spektrs, prominence=0.05, distance=3, width=1)
    plt.plot(peaks, spektrs[np.searchsorted(freq_range, peaks)], "o", label="Atrastie pīķi", alpha= 0.3)
    plt.legend()
    plt.show()

testejam(30, 30,20) #gausi un grādi


#plot(*(cetri_centri([35.4, 17.3,69.5])))