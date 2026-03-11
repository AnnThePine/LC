from main_func import cetri_centri, sign_dati, grad_vect, vect_grad, Vid_kvadr

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mainmag = np.linspace(0,10)

for i in mainmag:
    peaks =cetri_centri([59.578947368421055, 19.620689655172413,5],D=2875, tikaienergijas=True)
    print(peaks)