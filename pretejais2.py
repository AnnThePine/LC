import numpy as np
import matplotlib.pyplot as plt
import lmfit
import itertools

#freq range definēšana
amplituda = 6
bounds = (-amplituda, amplituda)

platums = 0.015

punkti = 5000

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
    eigenvalues = np.sort(eigen[0])
    return eigenvalues


def cetri_centri(lauka_kompoentes, griezums = "100"):
    # Ja lauka_kompoentes ir lmfit Parameters objekts, izvelkam vērtības
    if isinstance(lauka_kompoentes, lmfit.parameter.Parameters):
        lauka_kompoentes = np.array([
            lauka_kompoentes['Bx'].value,
            lauka_kompoentes['By'].value,
            lauka_kompoentes['Bz'].value])

    lauka_kompoentes = np.array(lauka_kompoentes)/ 1000

    energijas = []

    if griezums == "100":
        # 54.7gradi
        nvcentri = np.array([[-1,-1,1],
                    [1,1,1],
                    [-1,1,-1],
                    [1,-1,-1]])/np.sqrt(3)
        nvcentrix= np.array([np.cross(nvcentri[0,:],nvcentri[1,:]),
                              np.cross(nvcentri[0,:],nvcentri[1,:]),
                              np.cross(nvcentri[2,:],nvcentri[3,:]),
                              np.cross(nvcentri[2,:],nvcentri[3,:])])
        
        for i in range(4):
            nvcentrix[i,:] /= np.linalg.norm(nvcentrix[i,:])

        nvcentriy = np.array([np.cross(nvcentri[0,:],nvcentrix[0,:]),
                              np.cross(nvcentri[1,:],nvcentrix[1,:]),
                              np.cross(nvcentri[2,:],nvcentrix[2,:]),
                              np.cross(nvcentri[3,:],nvcentrix[3,:])])
        #jau ir vienības vect yey

        for NV in range(4):
            nvkomp = np.array([np.dot(nvcentrix[NV,:],lauka_kompoentes),
                               np.dot(nvcentriy[NV,:],lauka_kompoentes),
                               np.dot(nvcentri[NV,:],lauka_kompoentes)])
            en = ipasvertibas(*nvkomp)
            energijas.append(en[1]-en[0])
            energijas.append(en[2]-en[0])

    #print(energijas)
    mini = np.min(energijas)-0.2
    maxi = np.max(energijas)+0.2


    freq_range = np.linspace(mini, maxi, punkti)  # GHz
    odmr_signal = np.zeros_like(freq_range)

    for freq in energijas:
        gamma = platums / 2
        lorentz = -gamma**2 / ((freq_range - freq)**2 + gamma**2)
        odmr_signal += lorentz
    
    odmr_signal /= np.min(odmr_signal)

    return odmr_signal, freq_range

def grad_vect(alfa, beta):
    alfa = np.deg2rad(alfa)
    beta   = np.deg2rad(beta)

    x = np.sin(alfa) * np.cos(beta)
    y = np.sin(alfa) * np.sin(beta)
    z = np.cos(alfa)

    v = np.array([x, y, z], dtype=float)

    return v



def reverse(signal_data, grid_points=3, max_nfev=20000):
    """
    Combined version of reverese2 and reverse3.
    
    Parameters
    ----------
    signal_data : array-like
        Experimental/simulated spectrum to fit.
    amplituda : float
        Maximum search amplitude for Bx, By, Bz (mT).
    cetri_centri : callable
        Function that takes [Bx, By, Bz] and returns the spectrum (same length as signal_data).
    grid_points : int
        Number of points per axis for initial coarse grid search.
    max_nfev : int
        Maximum number of function evaluations for lmfit.
    
    Returns
    -------
    best_result : lmfit.MinimizerResult
        The best lmfit result object.
    best_params_list : list
        [Bx, By, Bz] of the best fit.
    """

    # Step 1: Initial coarse grid search
    bvals = np.linspace(-amplituda, amplituda, grid_points)
    start_grid = list(itertools.product(bvals, bvals, bvals))

    best_result = None
    best_resid = np.inf

    # Fitting helper function
    def residual_func(p):
        return cetri_centri([p['Bx'], p['By'], p['Bz']]) - signal_data

    for start_guess in start_grid:
        params = lmfit.Parameters()
        params.add('Bx', value=start_guess[0], min=-amplituda, max=amplituda)
        params.add('By', value=start_guess[1], min=-amplituda, max=amplituda)
        params.add('Bz', value=start_guess[2], min=-amplituda, max=amplituda)

        result = lmfit.minimize(
            residual_func,
            params,
            method="leastsq",
            max_nfev=max_nfev,
            xtol=1e-12, ftol=1e-12, gtol=1e-12
        )

        resid = np.sum(result.residual**2)
        if resid < best_resid:
            best_resid = resid
            best_result = result

    # Step 2: Refine using permutations of best absolute values with all sign combinations
        # Create parameter set from best_result
    best_params = best_result.params.valuesdict()

    # Use lmfit Parameters object with bounds
    params = lmfit.Parameters()
    params.add('Bx', value=best_params['Bx'], min=bounds[0], max=bounds[1])
    params.add('By', value=best_params['By'], min=bounds[0], max=bounds[1])
    params.add('Bz', value=best_params['Bz'], min=bounds[0], max=bounds[1])

    # Global optimization using differential evolution
    result = lmfit.minimize(
        residual_func,
        params,
        method='differential_evolution',
        max_nfev=max_nfev,
        tol=1e-10,
        updating='immediate'
    )

    resid = np.sum(result.residual**2)
    if resid < best_resid:
        best_resid = resid
        best_result = result

    
    # Return best result and its parameters as a list
    best_params_list = list(best_result.params.valuesdict().values())


    
    return best_result, best_params_list


def testejam(B, alfa, beta):
    mag_lauks = grad_vect(alfa, beta)*B # mT

    spektrs, freq_range = cetri_centri(mag_lauks)

    pretejais, pirma_rez = reverse(spektrs)

    print(mag_lauks)
    print(pirma_rez)


    plt.plot(freq_range, spektrs, label="simulated")
    simspektrs, simfreq = cetri_centri(pirma_rez)
    plt.plot(simfreq, simspektrs, label="model guess", alpha = 0.5)
    plt.legend()
    plt.show()

testejam(5, 60,30)