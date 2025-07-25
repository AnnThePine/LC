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



def viens_centrs(Virziens, no,lidz, lauksvajadzigs = False):

    mag_lauks = np.linspace(no,lidz, 100)

    lauks = np.empty((3,len(mag_lauks)))

        #lauks lin uz vektora komp
    lauks[0,:] = mag_lauks*Virziens[0]
    lauks[1,:] = mag_lauks*Virziens[1]
    lauks[2,:] = mag_lauks*Virziens[2]

    #print(lauks[:, 0])
    en, vectors = ipasvertibas(*lauks[:,0])

    energijas = np.empty((3,len(mag_lauks)))

    energijas[:,0] = en 

    for i in range(1, lauks.shape[1]):
        en, vect = ipasvertibas(*lauks[:,i])
        
        A = np.abs(vectors.T.conj()@vect)
        for e in range(3):

            energijas[e,i] = en[np.argmax(A[e, :])]

            vectors[:,e]=vect[:,np.argmax(A[e, :])]

    # print(energijas.shape)
    energiju_limeni = np.array((energijas[1]-energijas[0],energijas[2]-energijas[0]))

    if lauksvajadzigs:
        return mag_lauks, energiju_limeni
        #return mag_lauks, energijas

    else:    
        return energiju_limeni
        #return energijas



def plot(dati):
    plt.figure(figsize=[10,6])
    x,y = dati
    #print(x.shape, y.shape)

    style = ['-', (5, (10, 3)), (0, (3, 5, 1, 5, 1, 5)), ':']
    colors = [
    "#ff5500", 
    "#00EEEA",  
    "#6b0000",  
    "#005a18",  
    "#089E02", 
    "#BA05E7",  
    "#32ff39",  
    "#ff5ca0",  
    ]
    #print(y)
    if y[0].ndim == 1: 
        print("viendimencionāls")
        for en,i in enumerate(y):
            plt.plot(x,i, label = en)
    else:
        for nr,centrs in enumerate(y):
            for a,i in enumerate(centrs): 
                plt.plot(x,i, label = f"{nr+1} centrs, {a+1} enerģijas līmenis", linestyle = style[nr], linewidth=3, color = colors[2*nr+a])
    plt.tight_layout()
    plt.legend()
    plt.show()


#plot(viens_centrs([0,0,1],0,0.3,True))

def asis(z_axis):
    """Izveido lokālo (x,y,z), kur z = NV virziens."""
    z_axis = z_axis / np.linalg.norm(z_axis)  # NV normalizēts
    # Izvēlamies pagaidu vektoru, kas nav paralēls z
    tmp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    x_axis = np.cross(tmp, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack([x_axis, y_axis, z_axis])

def cetri_centri(lauka_virziens, no, līdz):


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
        en = viens_centrs(B_local, no, līdz)
        rezultati.append((en))

    lauks, en = viens_centrs(B_local, no, līdz, True)

    return lauks, rezultati


def grad_vect(alfa, beta):
    alfa = np.deg2rad(alfa)
    beta   = np.deg2rad(beta)

    x = np.sin(alfa) * np.cos(beta)
    y = np.sin(alfa) * np.sin(beta)
    z = np.cos(alfa)

    v = np.array([x, y, z], dtype=float)

    return v


plot(cetri_centri(grad_vect(90,35),0.0,0.3))


# lauks = np.empty((3,len(mag_lauks)))

#     alfa_rad = np.radians(alfa)
#     beta_rad = np.radians(beta)

#     lauks[0,:] = mag_lauks * np.sin(alfa_rad) * np.cos(beta_rad)
#     lauks[1,:] = mag_lauks * np.sin(alfa_rad) * np.sin(beta_rad)
#     lauks[2,:] = mag_lauks * np.cos(alfa_rad)