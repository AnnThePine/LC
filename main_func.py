import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy.signal import find_peaks
import mpmath as mp
import pandas as pd

amplituda = 20

platums = 1.5

punkti = 5000

mini = 2650
maxi = 3100

D = 2870.0  # MHz

Q = -5.1       # MHz
gamma_N = 0.3077e-3  # MHz/G   (0.3077 kHz/G)


Sx = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.complex128)/np.sqrt(2)
Sy = np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=np.complex128)/np.sqrt(2)
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=np.complex128)

Ix, Iy, Iz = Sx, Sy, Sz

I_n = np.eye(3, dtype=np.complex128)

g_bora = 2.803 # MHz/G 

alfa_const_dim = 1.06*10e-6#1/K

C = 1076+125-((2*125**2)/1076)

IzIz = Iz@Iz-2*I_n/3

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

# Ladder operators for spin-1
S_plus = np.array([[0, np.sqrt(2), 0],
                   [0, 0, np.sqrt(2)],
                   [0, 0, 0]], dtype=np.complex128)

S_minus = np.array([[0, 0, 0],
                    [np.sqrt(2), 0, 0],
                    [0, np.sqrt(2), 0]], dtype=np.complex128)

term1 = np.kron(S_plus, S_minus)
term2 = np.kron(S_minus, S_plus)

SIz = np.kron(Sz, Iz)
SI = np.kron(Sx, Ix) + np.kron(Sy, Iy)

mp.mp.dps = 60  # vari 50..100; 60 parasti pietiek 1e-12 stabilitātei

def eigvals_hp(H):
    A = mp.matrix([[mp.mpc(z.real, z.imag) for z in row] for row in H])
    vals = mp.eig(A)[0]                 # eigenvalues
    vals = sorted([mp.re(v) for v in vals])  # Herm
    return vals

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

def ipasvertibas(Bx, By, Bz,D, Mx, My, Mz ,agper, agparal,Q, Nx = 0, Ny = 0, nuclear = True, Precise = False):

    # electron zero-field splitting (MHz)
    SDS = D*Sz2

    # electron Zeeman (MHz)
    Zeeman  = g_bora * (Bz*Sz + Bx*Sx + By*Sy)

    # strain term (MHz)  --- new
    Strain = Mz*(Sz2)+Mx*Mxx+My*Myy+Nx*Nxx+Ny*Nyy

    # full electron part in full space (3 electron ⊗ 3 nuclear = 9x9)
    H_elec_full = SDS + Zeeman + Strain
    if nuclear:
    # nuclear-only part (MHz)
        H_nuc = - gamma_N * (Bx*Ix + By*Iy + Bz*Iz)+Q*Sz2

        H_nuc_full = np.kron(I_n, H_nuc)

    # hyperfine interaction (MHz)
        H_hyperfine = (agparal * SIz + agper   * SI)

        Hamiltonian = np.kron(H_elec_full, I_n) + H_nuc_full + H_hyperfine
    else:
        Hamiltonian = H_elec_full
    #print(Hamiltonian)
    Hamiltonian = np.array(Hamiltonian, dtype=np.complex128)
    Hamiltonian = (Hamiltonian + Hamiltonian.conj().T) / 2
    if Precise:
        eigenvalues = eigvals_hp(Hamiltonian)
    else: 
        eigenvalues = np.linalg.eigvalsh(Hamiltonian)
        eigenvalues= np.sort(eigenvalues)
    return eigenvalues

def Aproximated_stress(dT,b=7.1,a1=-11.7):
    Mx = -b*alfa_const_dim*C
    Mz = -a1*alfa_const_dim*C
    My = 0
    #Sz2+Sy2-Sx2
    return Mx, My, Mz

def Temperature_dependance(dT): #Dt0= 297K
    D = 2870.28-72.5*(10**-3)*dT-0.39*(10**-3)*(dT**2) #2870
    Q = -4945.88*(10**-3)+35.5*(10**-6)*dT+0.22*(10**-6)*(dT**2)
    Apar = -2165.19*(10**-3)+197*(10**-6)*dT+0.73*(10**-6)*(dT**2)
    Aper = -2635*(10**-3)+154*(10**-6)*dT+0.53*(10**-6)*(dT**2)
    return D,Q,Apar,Aper


def peak_to_starp(peaks):
    return [peaks[23]-peaks[2],peaks[20]-peaks[5],peaks[17]-peaks[8],peaks[14]-peaks[11]]

def Plane_stress_tensor(sxx, sxy, syy, a1=-11.7,a2=6.5,b=7.1,c=-5.4): 
    #Mhz/GPa(Barfuss), assumes plane stress, Nx, Ny not defined, needs further studies
    Mx = -b*(sxx +syy) + c*2*sxy
    My = np.sqrt(3)*b*(sxx - syy)
    Mz = a1*(sxx + syy) + 2*a2* sxy
    return Mx, My, Mz

def Stress_tensor(sxx,syy,szz,a1=-11.7,a2=6.5,b=7.1,c=-5.4, d = 2, e = 4, N_term  = False):
    sxy = np.sqrt(sxx**2+syy**2)
    sxz = np.sqrt(sxx**2+szz**2)
    syz = np.sqrt(syy**2+szz**2)

    Mx = b * (2*szz - sxx - syy) + c * (2*sxy - syz - sxz)
    My = np.sqrt(3) * b * (sxx - syy) + np.sqrt(3) * c * (syz - sxz)
    Mz = a1 * (sxx + syy + szz) + 2 * a2 * (syz + sxz + sxy)
    Nx = d * (2*szz - sxx - syy) + e * (2*sxy - syz - sxz)
    Ny = np.sqrt(3) * d * (sxx - syy) + np.sqrt(3) * e * (syz - sxz)

    if N_term:
        return Mx, My, Mz, Nx, Ny
    else:
        return Mx, My, Mz


def cetri_centri(sferiskas_koord, dT=0, Stress_dim_koord = [0,0,0],tpar=0,tper=0,tQ=0, vajagrange = True, griezums="100", tikaienergijas = False, FirstFit = False):

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

    D,Q,Apar,Aper = Temperature_dependance(dT)

    lauka_kompoentes = grad_vect(sferiskas_koord[0], sferiskas_koord[1])
    lauka_kompoentes = lauka_kompoentes*sferiskas_koord[2]
    all_energijas = []

    if griezums == "100":

        Mxes,Myes, Mzes = [], [],[]
        # Aprēķina enerģijas katram centram un apkopo kopā
        for NV in range(4):

            R_k = np.vstack([nvcentrix[NV], nvcentriy[NV], nvcentri[NV]]) #3*3 vect

            sigma_nv = [nvcentrix[NV]@ Stress_dim_koord, nvcentriy[NV]@Stress_dim_koord, nvcentri[NV]@Stress_dim_koord]

            Mx, My, Mz = Stress_tensor(*sigma_nv)

            Mxes.append(Mx)
            Myes.append(My)
            Mzes.append(Mz)

        for NV in range(4):

            nvkomp = np.array([
                np.dot(nvcentrix[NV, :], lauka_kompoentes),
                np.dot(nvcentriy[NV, :], lauka_kompoentes),
                np.dot(nvcentri[NV, :], lauka_kompoentes)])
            if FirstFit:
                en = np.array(ipasvertibas(*nvkomp,D, Mxes[NV],Myes[NV], Mzes[NV],agparal=tpar, agper=tper,Q=tQ))#agper=tper,Q=tQ
            else:
                en = np.array(ipasvertibas(*nvkomp,D, Mxes[NV],Myes[NV], Mzes[NV],agparal=Apar,agper=Aper,Q=Q))
            energijas = [en[8]-en[2],en[5]-en[2],
                 en[7]-en[1],en[3]-en[1],
                 en[6]-en[0],en[4]-en[0]]
            all_energijas.append(energijas)

        

    all_energijas = np.concatenate(all_energijas)

    #strain virzieni

    if tikaienergijas:
        ener=[]
        for e in all_energijas:
            if mini < e:
                ener.append(e)
        #ener = all_energijas

        if len(ener) == 0:
            print("kaut kas nogāja galīgi garām")
            return []

        # Sakārto pēc lieluma, lai vieglāk apvienot
        ener.sort()

        merged_peaks = []
        current_group = [ener[0]]

        for ee in ener[1:]:
            if abs(ee - current_group[-1]) <= 1.5:
                current_group.append(ee)
            else:
                avg = (min(current_group) + max(current_group)) / 2
                merged_peaks.append(avg)
                current_group = [ee]

        # pievieno pēdējo grupu
        if current_group:
            avg = (min(current_group) + max(current_group)) / 2
            merged_peaks.append(avg)

        return merged_peaks

    else:

        #maxi = np.max(all_energijas) + 20

        freq_range = np.linspace(mini, maxi, punkti)
        odmr_signal = np.zeros_like(freq_range)

        all_energijas = np.asarray(all_energijas, dtype=float).ravel()

        freq_range = np.linspace(float(mini), float(maxi), int(punkti)).astype(float)
        odmr_signal = np.zeros(freq_range.shape, dtype=float)

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
    
#fit functions
#temperature

def Temp_fit_function(params, coords,Apara):
    T  = params['T']
    Sx = params['Sx']
    Sy = params['Sy']

    Apar = Apara[0]
    Aper=Apara[1]
    q=Apara[2]
    
    S = [Sx, Sy, 0]
    res = cetri_centri(coords, dT=T,Stress_dim_koord=S,tpar=Apar, tper=Aper,tQ=q, tikaienergijas=True,FirstFit=True)
    return np.asarray(res)

def Temp_fit_residual_function(params, coords,Apara, real):
    pred = Temp_fit_function(params, coords,Apara)
    
    if len(pred) == 24:
        #print("yes")
        return (real - pred)**2
    else:
        #print(f"fuck \n {pred}")
        return np.full(24, 5000.0, dtype=float)


def Refining_fit(p, ts):
    #print(p)
    T  = ts[0]
    Sx = ts[1]
    Sy = ts[2]
    Apar = ts[3]
    Aper = ts[4]
    q = ts[5]
    
    coord = [p['alfa'],p['beta'],p['B']]
    
    S = [Sx, Sy, 0]
    res = cetri_centri(coord, dT=T,Stress_dim_koord=S,tpar=Apar, tper=Aper,tQ=q, tikaienergijas=True,FirstFit=True)
    
    return np.asarray(res)

def Refining_fit_residuals(p, ts, real):
    pred = Refining_fit(p, ts)
    
    if len(pred) == 24:
        #print("yes")
        return real - pred
    else:
        #print(f"fuck \n {pred}")
        return np.full(24, 5000.0, dtype=float)
    

matrix = np.array([[0, 1,1], [1, 1,1],[-1, 0,1],[0,0,1], [1, 0,1],[-1, -1,1],[0, -1,1], [1, -1,1],
              [-1, 1,0], [0, 1,0], [1, 1,0],[-1, 0,0],[0,0,0], [1, 0,0],[-1, -1,0], [0, -1,0], [1, -1,0],
              [-1, 1,-1], [0, 1,-1], [1, 1,-1],[-1, 0,-1],[0,0,-1], [1, 0,-1],[-1, -1,-1], [0, -1,-1], [1, -1,-1]])

params = lmfit.Parameters()
params.add('T',  value=40,   min=-10, max=100)
params.add('Sx', value=0,   min=-0.3, max=0.3)
params.add('Sy', value=0,   min=-0.3, max=0.3)

para = lmfit.Parameters()
para.add('Apar', value=-2.165,   min=-2.2, max=-1.8)
para.add('Aper', value=-2.635,   min=-2.7, max=-2.5)
para.add('Q', value=-4.95,   min=-4, max=-4.8)

def Apar_Fit(para, coords):
    Apar = para['Apar']
    Aper = para['Aper']
    q=para['Q']

    res = cetri_centri(coords,tpar=Apar, tper=Aper,tQ=q, tikaienergijas=True,FirstFit=True)
    return np.asarray(res)


def cheating(dat):
    starp1=[]
    starp2=[]
    for i in range(0,5):
        starp1.append(dat[i+3]-dat[i+1])
        starp2.append(dat[i+3]-dat[i+2])
    return np.average(starp1),np.average(starp2)

def Apar_Fit_res(para, coords, real):
    pred = Apar_Fit(para, coords)
    
    if len(pred) == 24:
        r1, r2 = cheating(real)
        p1, p2 = cheating(pred)
        return 10.0 * ((r1 - p1)**2 + (r2 - p2)**2) 
    else:
        #print(f"fuck \n {pred}")
        return np.full(2, 50000000.0, dtype=float)



def meklebias(peaks,field_lookup_df):
    starppeaks = peak_to_starp(peaks)

    field_lookup_df["err"] = field_lookup_df["values"].apply(
        lambda row: np.sum(
            Vid_kvadr(np.array(row[:len(starppeaks)]), np.array(starppeaks))
        )
    )

    sorted_df = field_lookup_df.sort_values(by="err", ascending=True).reset_index(drop=True)

    vert = []
    kludinas = []

    for i in range(0,10):
        coooo = sorted_df.iloc[i].copy()
        gridmin = np.array(coooo["coordinates"])
        att = 2
        kluda = 0
        print(i)
        for a in range(15): 
            points = gridmin + matrix * att
            kludas = []
            for p in points:
                energ = sign_dati(*cetri_centri(p, tikaienergijas=False, vajagrange=True))[0]
                if len(energ) == 24:
                    starpe = peak_to_starp(energ)
                    kludas.append(np.sum(Vid_kvadr(starpe, starppeaks)))
                else:
                    kludas.append(np.inf)

            # find best point index
            best_idx = np.argmin(kludas)
            best_point = points[best_idx]
            best_error = kludas[best_idx]

            # update gridmin ONCE
            if best_error <= 10 * att:
                gridmin = best_point
                att /= 2
            else:
                print(f"apstajas pie: {abs(np.log2(att)) + 1}")
                beigu = att
                break
        vert.append(gridmin)
        kludinas.append(best_error)
    arr = np.column_stack([np.asarray(vert), np.asarray(kludinas)])  # shape (n, 4)
    # columns: [alfa, beta, B, err]

    arr_sorted = arr[np.argsort(arr[:, -1])] 

    
    finall = []

    for aaa in range(5):
        rough = arr_sorted[aaa][:3]
        ree = lmfit.minimize(Apar_Fit_res, para,args=(rough, peaks),method='differential_evolution')
        Apara = [ree.params[name].value for name in ree.var_names]
        rez = lmfit.minimize(Temp_fit_residual_function, params, args=(rough,Apara, peaks),method='differential_evolution') 
        print(rez.params)
        tesrt = [rez.params[name].value for name in rez.var_names]
        #print (tesrt) #[22.211847912061096, -0.008420355573983851, 0.018460267821950704]

        tesrt.append(Apara[0])
        tesrt.append(Apara[1])
        tesrt.append(Apara[2])

        p = lmfit.Parameters()
        p.add('alfa', value=rough[0],min= rough[0]-1,max= rough[0]+1)
        p.add('beta', value=rough[1],min= rough[1]-1,max= rough[1]+1)
        p.add('B',value=rough[2],min= rough[2]-2,max= rough[2]+2)

        end_e = lmfit.minimize(Refining_fit_residuals, p, args=(tesrt, peaks))

        finpar = [end_e.params[name].value for name in end_e.var_names]

        # print(np.sqrt(np.sum(end_e.residual**2)))

        # print(finpar)
        finall.append([*finpar,*tesrt, np.sqrt(np.sum(end_e.residual**2))])
    finarray = np.asarray(finall)
    arrrrrr = finarray[finarray[:, -1].argsort()]
    return arrrrrr[0]



def Read_init(data_name, lookup_name,minalfa = 5,maxalfa = 85,minB = 5,maxB = 500):
    # Read your CSV
    try:
        ielasitais = pd.read_csv(lookup_name)
        testa_dati = pd.read_csv(data_name)
    except FileNotFoundError:
        print("Lookup table not found. Please run the generation script first.")
        exit()



    # Apply both filters in one vectorized operation
    ielasitais = ielasitais[
        (ielasitais['alfa'] >= minalfa) & (ielasitais['alfa'] <= maxalfa) &
        (ielasitais['B'] >= minB) & (ielasitais['B'] <= maxB)
    ].reset_index(drop=True)

    # Combine the first three columns into a tuple or string
    # Example: tuple
    coords = list(zip(
        ielasitais.iloc[:, 0],  # alfa
        ielasitais.iloc[:, 1],  # beta
        ielasitais.iloc[:, 2]   # B
    ))

    # Collect the rest of the columns into lists
    values = ielasitais.iloc[:, 3:].apply(lambda row: row.dropna().tolist(), axis=1).tolist()

    # Build the new DataFrame
    field_lookup_df = pd.DataFrame({
        'coordinates': coords,
        'values': values
    })

    print("sākas bias meklēšana")

    return testa_dati,field_lookup_df
