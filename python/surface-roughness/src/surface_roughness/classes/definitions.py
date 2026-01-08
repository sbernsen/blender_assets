import numpy as np 
import matplotlib.pyplot as plt 

import gstools as gs 
from gstools.field.generator import Fourier


# ================================== Surfaces ==================================
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

# Spectral field
def spectral_field(variance, length_scale, dim = 2):
    model = gs.Gaussian(dim=2, var=0.0225, len_scale=2.0)  # start isotropic
    gen = Fourier(model, period=[np.ptp(x), np.ptp(y)], mode_no=512, seed=42)
    srf = gs.SRF(
        model,
        generator="Fourier",               # <- select the Fourier generator
        period=[np.ptp(x), np.ptp(y)],         # domain size per axis
        mode_no=[512, 512],                # modes per axis (or a single int)
        seed=42,
    )
    z = srf.structured((x, y))   



# ============================ 3D Domain Generation ============================
# ------------------------------------------------------------------------------
def extrude_domain_3d(geometry_array: np.ndarray, ny: int) -> np.ndarray:
    """
    
    
    """
    geometry_array = np.asarray(geometry_array)
    if geometry_array.ndim != 2:
        rase ValueError("Geometry must be a 2-D array of shape (nz, nx).")
    
    nz, nx = geometry_array.shape 
    labels = np.tile(geoemtry_array[:,None, :], (1, ny, 1)) # (nz, ny, nx)
    return labels
    

# ------------------------------------------------------------------------------
def voxelize_surface(
        geometry_3d, 
        surface_model,
        *,
        id: int,
        vertical_thickness = 1, 
        vertical_shift: int = 0,
        mode: str = "below",
        clamp: bool = True
    ):
    '''
    '''
    if geometry_3d.ndim != 3:
        raise ValueError("geometry_3d must be a 3-D array shaped (nz, ny, nx).")
    nz, ny, nx = geometry_3d.shape

    surface_model = np.asarray(surface_model)
    if surface_model.shape != (ny, nx):
        raise ValueError(
            f"surface_model must have shape (ny, nx)=({ny}, {nx}), "
            f"but got {surface_model.shape}."
        )
    if vertical_thickness < 1:
        raise ValueError("vertical_thickness must be >= 1.")

    # Center indices (rounded) at each (y, x)
    k_center = np.rint(surface_model).astype(np.int64) + int(vertical_shift)

    # Compute per-(y,x) start/stop indices along z based on the mode
    T = int(vertical_thickness)
    if mode == "below":
        k0 = k_center
        k1 = k_center + T
    elif mode == "above":
        k0 = k_center - (T - 1)
        k1 = k_center + 1
    elif mode == "two-sided":
        half_lo = T // 2
        half_hi = T - half_lo
        k0 = k_center - half_lo
        k1 = k_center + half_hi
    else:
        raise ValueError("mode must be 'below', 'above', or 'two-sided'.")

    if clamp:
        k0 = np.clip(k0, 0, nz)   # allow k0==nz (empty slice)
        k1 = np.clip(k1, 0, nz)
    else:
        if (k0 < 0).any() or (k1 > nz).any():
            raise IndexError("Surface layer extends outside z bounds.")

    # Build a boolean mask in a fully vectorized way
    Z = np.arange(nz, dtype=np.int64)[:, None, None]        # (nz,1,1)
    start = k0[None, :, :]                                  # (1,ny,nx)
    stop  = k1[None, :, :]                                  # (1,ny,nx)
    layer_mask = (Z >= start) & (Z < stop)                  # (nz,ny,nx), True where we fill

    # Write into a copy (so caller can choose to keep original)
    out = np.array(geometry_3d, copy=True)
    out[layer_mask] = int(id)
    return out

