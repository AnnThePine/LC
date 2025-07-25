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
bora = 14

def ipasvertibas(Bx, By, Bz, D=2.88):

    Dzz = D * 2 / 3
    Dxx = Dyy = -Dzz / 2  # From Dxx + Dyy + Dzz = 0
    SDS = Dxx * (Sx @ Sx) + Dyy * (Sy @ Sy) + Dzz * (Sz @ Sz)
    Hamitonis = SDS + (g * bora *(Bz * Sz + Bx*Sx + By*Sy))

    eigen = np.linalg.eig(Hamitonis)
    #print(Bz,Bx)
    print(eigen[0])
    print(eigen[1])
    print(eigen[1].T.conj()@eigen[1])
    eigenvalues = eigen[0]#.real
    eigenvectors = eigen[1]#.real
    

    # for eing in range(3):
    #     eigenvectors[:,eing]*=np.sign(eigenvectors[0,eing])


    #print(eigenvalues)
    #print(eigenvectors)
    #print(eigenvectors.T@eigenvectors)
    return eigenvalues[0], eigenvalues[1], eigenvalues[2], eigenvectors


def lauksUnEnergijas(B,alfa,beta):
    alfa_rad = np.radians(alfa)
    beta_rad = np.radians(beta)

    x = B * np.sin(alfa_rad) * np.cos(beta_rad)
    y = B * np.sin(alfa_rad) * np.sin(beta_rad)
    z = B * np.cos(alfa_rad)

    i0,i1,i2, ipasvectori = ipasvertibas(x,y,z)

    return [i0,i1,i2], ipasvectori

#def kartoshana():



def viens_centrs(alfa,beta, no,lidz, lauksvajadzigs = False):

    lauks = np.linspace(no,lidz, 100)
    
    
    en, vectors = lauksUnEnergijas(lauks[0], alfa, beta)

    energijas = np.empty((3,len(lauks)))

    energijas[:,0] = en

    

    for i, B in enumerate(lauks[1:],1):
        en, vect = lauksUnEnergijas(B, alfa, beta)
        
        A = np.abs(vectors.T.conj()@vect)
        for e in range(3):
            # #vektors
            # skalarie_reiz = []

            # for z in range(3):
            #     #vect
            #     skalarie_reiz.append(np.dot(vectors[:,e],vect[:,z]))
            #     #skalarie_reiz.append(np.linalg.norm(vectors[:,e]-vect[:,z]))
            
            # energijas[e,i] = en[np.argmax(skalarie_reiz)]
            # print(skalarie_reiz)

            energijas[e,i] = en[np.argmax(A[e, :])]
            # print(np.argmax(A[e, :]))

            vectors[:,e]=vect[:,np.argmax(A[e, :])]

        #print(A)
        # vectors = vect

            #print(eigenvectors.T@eigenvectors)
            # energijas[e,i] = en[np.argmax(np.abs(vect[:,e]))]

    
    #print(energijas)

    energiju_limeni = np.array((energijas[0]-energijas[1],energijas[2]-energijas[1]))
    #print(energiju_limeni)

    if lauksvajadzigs:
        return lauks, energiju_limeni
        # return lauks, energijas

    else:    
        return energiju_limeni
        # return energijas
    

def plot(dati):
    plt.figure(figsize=[10,6])
    x,y = dati
    print(x.shape, y.shape)
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
        for en,i in enumerate(y):
            plt.plot(x,i, label = en)
    else:
        for nr,centrs in enumerate(y):
            for a,i in enumerate(centrs): 
                plt.plot(x,i, label = f"{nr+1} centrs, {a+1} enerģijas līmenis", linestyle = style[nr], linewidth=3, color = colors[2*nr+a])
    plt.tight_layout()
    plt.legend()
    plt.show()



def cetri_centri(alfa, beta, no, līdz):
    #     #pirmais vnk tāpat
    lauks, en1 = viens_centrs(alfa,beta, no, līdz,True)

#     #otrais ybeta=0,x alfa-109.5
    en2 = viens_centrs(alfa-109.5, beta, no, līdz)

#     # treshais, ceturtais alfa-109.5, beta _+120
#     #treshais
    en3 = viens_centrs(alfa-109.5, beta+120,no, līdz)

#     #ceturtais
    en4 = viens_centrs(alfa-109.5, beta-120,no, līdz)

    energijas = [en1,en2,en3,en4]
    return lauks, energijas
    
    


# plot(cetri_centri(0,0,0,0.3))
plot(viens_centrs(5,0,0,0.3, True))

