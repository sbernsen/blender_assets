import numpy as np 
import matplotlib.pyplot as plt 

import gstools as gs 
from gstools.field.generator import Fourier

# Gaussian field
def gaussian_field(variance, length_scale, dim = 2, angles = 0):
    '''
    
    '''
    model = gs.Gaussian(
        dim=dim, 
        var = variance, 
        len_scale = length_scale, 
        angles = angles
    )
    srf = gs.SRF(model, seed=42)
    x = np.linspace(0, 128, 512 )
    y = x.copy() 
    X, Y = np.meshgrid(x, y, indexing = "xy")
    z = srf((X.ravel(), Y.ravel()), mesh_type="unstructured").reshape(Y.shape)
    
    return z



