import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy.signal import find_peaks
import pandas as pd

FWHM = 1.5
Plot_points = 5000

dat = "merged_data1.csv"
look = "Starp_lookup_table.csv"

Min_MHz = 2100
Max_MHz = 3100

Gamma_N = 0.3077e-3  # MHz/G   (0.3077 kHz/G)

Bhor_magnethron = 2.803 # MHz/G 

Diamond_termal_exspantion_coef = 1.06*10e-6#1/K

C = 1076+125-((2*125**2)/1076)

Sx = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.complex128)/np.sqrt(2)
Sy = np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=np.complex128)/np.sqrt(2)
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]], dtype=np.complex128)

Ix, Iy, Iz = Sx, Sy, Sz
Iz2 = Iz @ Iz

I_n = np.eye(3, dtype=np.complex128)

Sx2 = Sx @ Sx
Sy2 = Sy @ Sy
Sz2 = Sz @ Sz

Mxx = Sx2-Sy2
Myy = Sx@Sy + Sy@Sx
Nxx = Sx@Sz + Sz@Sx
Nyy = Sy@Sz + Sz@Sy

term1 = np.kron(Sx + 1j*Sy, Sx - 1j*Sy)
term2 = np.kron(Sx - 1j*Sy, Sx + 1j*Sy)

SIz = np.kron(Sz, Iz)
SI  = np.kron(Sx, Ix) + np.kron(Sy, Iy)


nvcentri = np.array([[-1, -1, 1],
                        [1, 1, 1],
                        [-1, 1, -1],
                        [1, -1, -1]]) / np.sqrt(3)

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

Zeroth = {
    "Dplus":[2870,2870,2870,2870],
    "Dminus":[2870,2870,2870,2870],
    "stress_dim":[0,0,0],
    "par":[-2.16,-2.19,-2.12,-2.2],
    "per":[-3,-3,-3,-3],
    "Q":[-4.45,-4.41,-3.77,-3.66]
}

def Eigenvalues(Bx, By, Bz, Dplus, Dminus, Mx, My, Mz ,agper, agparal, Q, Nx = 0, Ny = 0, nuclear = True):

    # electron zero-field splitting (MHz)
    D= np.array([[Dplus,0,0],[0,0,0],[0,0,Dminus]])

    # electron Zeeman (MHz)
    Zeeman  = Bhor_magnethron * (Bz*Sz + Bx*Sx + By*Sy)

    # strain term (MHz)  --- new
    Strain = Mz*(Sz2)+Mx*Mxx+My*Myy+Nx*Nxx+Ny*Nyy

    # full electron part in full space (3 electron ⊗ 3 nuclear = 9x9)
    H_elec_full = D + Zeeman + Strain
    if nuclear:
    # nuclear-only part (MHz)
        H_nuc = -Gamma_N*(Bx*Ix + By*Iy + Bz*Iz) + Q*Iz2

        H_nuc_full = np.kron(I_n, H_nuc)

    # hyperfine interaction (MHz)
        H_hyperfine = (agparal * SIz + agper   * SI)

        Hamiltonian = np.kron(H_elec_full, I_n) + H_nuc_full + H_hyperfine
    else:
        Hamiltonian = H_elec_full
    #print(Hamiltonian)
    Hamiltonian = np.array(Hamiltonian, dtype=np.complex128)
    Hamiltonian = (Hamiltonian + Hamiltonian.conj().T) / 2
    
    eigenvalues = np.linalg.eigvalsh(Hamiltonian)
    eigenvalues= np.sort(eigenvalues)

    return eigenvalues


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
    
Grid_matrix = np.array([[0, 1,1], [1, 1,1],[-1, 0,1],[0,0,1], [1, 0,1],[-1, -1,1],[0, -1,1], [1, -1,1],
              [-1, 1,0], [0, 1,0], [1, 1,0],[-1, 0,0],[0,0,0], [1, 0,0],[-1, -1,0], [0, -1,0], [1, -1,0],
              [-1, 1,-1], [0, 1,-1], [1, 1,-1],[-1, 0,-1],[0,0,-1], [1, 0,-1],[-1, -1,-1], [0, -1,-1], [1, -1,-1]])

def Grad_vect(alfa, beta):
    alfa = np.deg2rad(alfa)
    beta   = np.deg2rad(beta)

    x = np.sin(alfa) * np.cos(beta)
    y = np.sin(alfa) * np.sin(beta)
    z = np.cos(alfa)

    v = np.array([x, y, z], dtype=float)

    return v

Lorentz = lambda f, A, fwhm, f0 :A*(fwhm / 2)**2 / ((f - f0)**2 + (fwhm / 2)**2)

def Peaks_to_diference(peaks):
    return [peaks[23]-peaks[2],peaks[20]-peaks[5],peaks[17]-peaks[8],peaks[14]-peaks[11]]

def cetri_centri(sferiskas_koord, params = Zeroth, cut="100"):

    #reading necesary parameters
    Par = params["par"]
    Per = params["per"]
    Qq=params["Q"]
    Dplus=params["Dplus"]
    Dminus=params["Dminus"]
    Stress_dim_koord = params["stress_dim"]

    
    # if lauka_kompoentes is lmfit Parameters 
    if isinstance(sferiskas_koord, lmfit.parameter.Parameters):
        sferiskas_koord = np.array([
            sferiskas_koord['Bvert'].value,
            sferiskas_koord['Bhor'].value,
            sferiskas_koord['Babs'].value])
        
    sferiskas_koord = np.array(sferiskas_koord)

    # Safety if sferiskas_koord arent 3 values
    if sferiskas_koord.size < 3:
        raise ValueError(f"sferiskas_koord jāsatur 3 vērtības (vert, hor, abs). Saņemts: {sferiskas_koord}")

    lauka_kompoentes = Grad_vect(sferiskas_koord[0], sferiskas_koord[1])
    lauka_kompoentes = lauka_kompoentes*sferiskas_koord[2]
    all_energijas = []

    if cut == "100":
        Mxes,Myes, Mzes = [], [],[]

        # Cauculates energies for each center
        for NV in range(4):

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
            
            en = np.array(Eigenvalues(*nvkomp,Dplus[NV],Dminus[NV], Mxes[NV],Myes[NV], Mzes[NV],agparal=Par[NV], agper=Per[NV],Q=Qq[NV]))
            energijas = [en[8]-en[2],en[5]-en[2],
                 en[7]-en[1],en[3]-en[1],
                 en[6]-en[0],en[4]-en[0]]
            all_energijas.append(energijas)

    all_energijas = np.concatenate(all_energijas)

    if len(all_energijas) != 24:
        print("kaut kas nogāja galīgi garām")
        return []

    # Sakārto pēc lieluma, lai vieglāk apvienot
    all_energijas.sort()

    merged_peaks = [all_energijas[0]]

    for ee in all_energijas[1:]:
        if abs(ee - merged_peaks[-1])<= FWHM:
            avg = (merged_peaks[-1] + ee) / 2
            merged_peaks.pop()
            merged_peaks.append(avg)
        else:
            merged_peaks.append(ee)
    return merged_peaks

def Average(x,y,square = True):
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

def Refining_fit(p):

    coord = [p['alfa'],p['beta'],p['B']]

    reff = {
        "Dplus":[p["Dplus1"],p["Dplus2"],p["Dplus3"],p["Dplus4"]],
        "Dminus":[p["Dminus1"],p["Dminus2"],p["Dminus3"],p["Dminus4"]],
        "stress_dim":[p['Sx'],p['Sy'],0],
        "par":[p['Apar1'],p['Apar2'],p['Apar3'],p['Apar4']],
        "per":[p['Aper1'],p['Aper2'],p['Aper3'],p['Aper4']],
        "Q":[p['Q1'],p['Q2'],p['Q3'],p['Q4']]
    }

    res = cetri_centri(coord, params=reff, tikaienergijas=True)
    
    return np.asarray(res)

def Fit(p, real):
    pred = Refining_fit(p)
    
    if len(pred) == 24:
        #print("yes")
        return real - pred
    else:
        #print(f"fuck \n {pred}")
        return np.full(24, 5000.0, dtype=float)

def Find_bias(peaks,field_lookup_df):
    starppeaks = Peaks_to_diference(peaks)

    field_lookup_df["err"] = field_lookup_df["values"].apply(
        lambda row: np.sum(Average(np.array(row[:len(starppeaks)]), np.array(starppeaks)))
        )

    sorted_df = field_lookup_df.sort_values(by="err", ascending=True).reset_index(drop=True)

    vert = []
    kludinas = []

    for i in range(0,10):
        coooo = sorted_df.iloc[i].copy()
        gridmin = np.array(coooo["coordinates"])
        att = 2
        print(i)
        for a in range(15): 
            points = gridmin + Grid_matrix * att
            kludas = []
            for p in points:
                energ = cetri_centri(p)
                if len(energ) == 24:
                    starpe = Peaks_to_diference(energ)
                    kludas.append(np.sum(Average(starpe, starppeaks)))
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
                break
        vert.append(gridmin)
        kludinas.append(best_error)
    arr = np.column_stack([np.asarray(vert), np.asarray(kludinas)])  # shape (n, 4)
    # columns: [alfa, beta, B, err]

    arr_sorted = arr[np.argsort(arr[:, -1])] 

    finall = []

    for aaa in range(5):

        rough = arr_sorted[aaa][:3]

        params = lmfit.Parameters()
        params.add('alfa', value=rough[0],min= rough[0]-1,max= rough[0]+1)
        params.add('beta', value=rough[1],min= rough[1]-1,max= rough[1]+1)
        params.add('B',value=rough[2],min= rough[2]-2,max= rough[2]+2)
        params.add('Dplus1',  value=2870,min=2840, max=2900)
        params.add('Dminus1',  value=2870,min=2840, max=2900)
        params.add('Dplus2',  value=2870,min=2840, max=2900)
        params.add('Dminus2',  value=2870,min=2840, max=2900)
        params.add('Dplus3',  value=2870,min=2840, max=2900)
        params.add('Dminus3',  value=2870,min=2840, max=2900)
        params.add('Dplus4',  value=2870,min=2840, max=2900)
        params.add('Dminus4',  value=2870,min=2840, max=2900)
        params.add('Sx', value=0,   min=-0.3, max=0.3)
        params.add('Sy', value=0,   min=-0.3, max=0.3)
        params.add('Apar1', value=-2.12,   min=-2.2, max=-2)
        params.add('Aper1', value=-2.635,   min=-2.7, max=-3)
        params.add('Apar2', value=-2.12,   min=-2.2, max=-2)
        params.add('Aper2', value=-2.635,   min=-2.7, max=-3)
        params.add('Apar3', value=-2.12,   min=-2.2, max=-2)
        params.add('Aper3', value=-2.635,   min=-2.7, max=-3)
        params.add('Apar4', value=-2.12,   min=-2.2, max=-2)
        params.add('Aper4', value=-2.635,   min=-2.7, max=-3)
        params.add('Q1', value=-4.95,   min=-5, max=-3.3)
        params.add('Q2', value=-4.95,   min=-5, max=-3.3)
        params.add('Q3', value=-4.95,   min=-5, max=-3.3)
        params.add('Q4', value=-4.95,   min=-5, max=-3.3)



        for p in params: params[p].vary = False
        params['Dplus2'].vary =params['Dminus2'].vary=params['Dplus4'].vary =params['Dminus4'].vary=params['Dplus3'].vary =params['Dminus3'].vary=params['Dplus1'].vary =params['Dminus1'].vary=True
        rezy = lmfit.minimize(Fit, params, args=(peaks,),method='differential_evolution') 
        params = rezy.params


        for p in params: params[p].vary = False
        params['Sx'].vary= params['Sy'].vary=True
        rez = lmfit.minimize(Fit, params, args=(peaks,),method='differential_evolution') 
        params = rez.params
        

        for p in params: params[p].vary = False
        params['Apar1'].vary = params['Aper1'].vary=params['Apar2'].vary = params['Aper2'].vary=params['Apar3'].vary = params['Aper3'].vary= params['Apar4'].vary = params['Aper4'].vary=params['Q1'].vary=params['Q2'].vary=params['Q3'].vary=params['Q4'].vary=True
        ree = lmfit.minimize(Fit, params,args=(peaks,),method='differential_evolution')
        params = ree.params

        for p in params: params[p].vary = False
        params['alfa'].vary = params['beta'].vary= params['B'].vary=True
        end_e = lmfit.minimize(Fit, params, args=(peaks,))
        params = end_e.params

        # finpar = [params[p].value for p in params]
        # print(np.sqrt(np.sum(end_e.residual**2)))

        # print(finpar)
        finall.append({**params, 'residual': np.sqrt(np.sum(end_e.residual**2))})
    #finarray = np.asarray(finall)
    arrrrrr = sorted(finall, key=lambda x: x['residual'])
    return arrrrrr[0]

















