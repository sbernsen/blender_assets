import numpy as np 
import matplotlib.pyplot as plt 

import gstools as gs 
from gstools.field.generator import Fourier

# Gaussian field
model = gs.Gaussian(dim=2, var = 0.0225, len_scale = [2.0, 1.0], angles = 45.0)
srf = gs.SRF(model, seed=42)
x = np.linspace(0, 128, 512 )
y = x.copy() 
X, Y = np.meshgrid(x, y, indexing = "xy")
z = srf((X.ravel(), Y.ravel()), mesh_type="unstructured").reshape(Y.shape)

# --- Plot the surface ---
extent = (x.min(), x.max(), y.min(), y.max())

fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (13, 6) )
im = ax1.imshow(z, origin="lower", extent=extent, aspect="equal")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
cbar = plt.colorbar(im)
cbar.set_label("Height z [m]")
ax1.tight_layout()

# --- (Optional) Plot slope magnitude (relevant for scattering) ---
dzdx, dzdy = np.gradient(z, x, y)   # account for grid spacing
slope = np.hypot(dzdx, dzdy)

im2 = ax2.imshow(slope, origin="lower", extent=extent, aspect="equal")
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_title("Surface Slope Magnitude |∇z|")
cbar2 = plt.colorbar(im2)
cbar2.set_label("unitless")

ax2.tight_layout()
plt.show()


# Spectral field 
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

extent = (x.min(), x.max(), y.min(), y.max())

fig, (ax1s, ax2s) = plt.subplots(ncols = 2, figsize = (13, 6) )
im = ax1s.imshow(z, origin="lower", extent=extent, aspect="equal")
ax1s.set_xlabel("x [m]")
ax1s.set_ylabel("y [m]")
cbars = plt.colorbar(im)
cbars.set_label("Height z [m]")
# ax1s.tight_layout()

# --- (Optional) Plot slope magnitude (relevant for scattering) ---
dzdx, dzdy = np.gradient(z, x, y)   # account for grid spacing
slope = np.hypot(dzdx, dzdy)

im2 = ax2s.imshow(slope, origin="lower", extent=extent, aspect="equal")
ax2s.set_xlabel("x [m]")
ax2s.set_ylabel("y [m]")
ax2s.set_title("Surface Slope Magnitude |∇z|")
cbar2s = plt.colorbar(im2)
cbar2s.set_label("unitless")

plt.tight_layout()
plt.show()