# def letstrythisshit(alfa, beta, mag,  Print = False, graph = False):
#     lauks = [alfa, beta, mag]

#     lauksvec = grad_vect(alfa,beta)*mag

#     freq, odmr = cetri_centri(lauks)

#     peaks, peaky = sign_dati(freq, odmr)

#     #print(peaks)

#     reeee = meklebias(peaks)
#     res = (reeee[0],reeee[1],reeee[2])

    
#     paramlauks = grad_vect(res[0],res[1])*res[2]

#     starp1 = []
#     for i in range(3):
#         starp1.append(np.abs((lauksvec[i]-paramlauks[i])*100000)) #100000


#     if Print:
#         print(f"īstais lauks: {lauks}")
#         print(f"rezultats: {res}")
#         print(f"starpiba(nT): {starp1}") 
#         #print(f"Laiks: {laiks}")

#     if graph:
#         modfreq, mododmr = cetri_centri(res)

#         plt.plot(freq, odmr, label="simulated")
#         plt.plot(modfreq, mododmr, label = "guessed")
#         plt.scatter(peaks, peaky)
#         plt.legend()
#         plt.show()

#     return([[alfa, beta, mag],res, starp1])


# def sweep(graf = False):
#     alfas = np.linspace(minalfa, maxalfa, 40)

#     starpibas1 = []
#     starpibas2 = []
#     starpibas3 = []
#     laiki = []
#     for beta in alfas:
#         _,_,starp,laiks = letstrythisshit(beta,25,30, Print=True, graph = False)
#         starpibas1.append(starp[0])
#         starpibas2.append(starp[1])
#         starpibas3.append(starp[2])
#         #print(starpibas1)
#         laiki.append(laiks)

#     if graf:
#         fig, (ax1, ax2) = plt.subplots(2)
#         ax1.plot(alfas, starpibas1, label = "x")
#         ax1.plot(alfas, starpibas2, label = "y")
#         ax1.plot(alfas, starpibas3, label = "z")
#         ax1.legend()
#         ax1.set_xlabel("alfa leņķis, grad")
#         ax1.set_ylabel("mag. lauka kļūdas nanoteslās")
#         ax2.plot(alfas, laiki)
#         ax2.set_xlabel("alfa leņķis, grad")
#         ax2.set_ylabel('laiks, sec')

#         plt.tight_layout()
#         plt.show()
    
#     return(max(starpibas1),max(starpibas2),max(starpibas3))