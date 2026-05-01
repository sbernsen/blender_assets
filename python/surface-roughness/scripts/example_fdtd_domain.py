#!/usr/bin/env python3
"""
End-to-end example: generate rough surfaces, build a 3-D FDTD domain,
and prepare inputs for the SeidarT solver.

Usage
-----
    conda activate seidart
    python example_fdtd_domain.py

This script demonstrates:
1. Creating a base 3-D geometry (two-material half-space)
2. Generating multiple rough surfaces with independent statistics
3. Inserting them (with optional tilt) into the domain
4. Writing ``geometry.dat`` and updating a SeidarT project JSON
5. Visualising cross-sections of the result

The generated domain can then be loaded by the standard SeidarT pipeline::

    from seidart.routines.definitions import loadproject
    from seidart.routines.classes import Domain, Material, Model
    domain, material, seismic, em = loadproject(
        'rough_surface_project.json',
        Domain(), Material(), Model(), Model()
    )
    seismic.build(material, domain)
    seismic.run()
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the package to the path so we can import without installing
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', 'src')
)

from surface_roughness import (
    RoughSurface,
    gaussian_field,
    spectral_field,
    height_to_indices,
    validate_resolution,
    extrude_domain_3d,
    voxelize_surface,
    insert_surface_rotated,
    write_geometry,
    build_seidart_surfaces,
)

# ============================================================================
# Domain parameters
# ============================================================================
nx, ny, nz = 200, 100, 120   # grid points (SeidarT convention: x, y, z)
dx = dy = dz = 0.25          # grid spacing in metres
lx = nx * dx                 # physical length x
ly = ny * dy                 # physical length y
lz = nz * dz                 # physical length z

# Material IDs
AIR   = 0
ROCK  = 1
ICE   = 2
WATER = 3

# ============================================================================
# Step 1: Build a base 2-D cross-section and extrude to 3-D
# ============================================================================
# Simple half-space: air on top, rock on bottom
cross_section = np.zeros((nx, nz), dtype=int)
cross_section[:, nz // 2:] = ROCK  # bottom half is rock

geometry = extrude_domain_3d(cross_section, ny)
print(f"Base geometry shape (nx, ny, nz): {geometry.shape}")

# ============================================================================
# Step 2: Generate rough surfaces
# ============================================================================

# Surface 1 — rough bedrock interface (axis-aligned, no rotation)
surf_bedrock = RoughSurface(
    name="rough_bedrock",
    material_id=ICE,
    rgb="128/200/255",
    variance=4.0,              # σ² in metres²
    length_scale=[5.0, 3.0],   # anisotropic correlation lengths
    method="gaussian",
    angles=30.0,               # anisotropy orientation
    seed=42,
    reference_point=(0.0, 0.0, nz // 2 * dz),  # centre of domain in z
    vertical_thickness=8,      # 8 cells (= 2 m) thick
    mode="two-sided",
)
surf_bedrock.generate(nx, ny, lx, ly)

# Surface 2 — a tilted ice layer (rotated 10° around x-axis)
surf_tilted = RoughSurface(
    name="tilted_ice_layer",
    material_id=WATER,
    rgb="0/0/255",
    variance=1.0,
    length_scale=4.0,
    method="gaussian",
    seed=99,
    reference_point=(0.0, 0.0, (nz // 2 + 15) * dz),
    angle_x=10.0,              # 10° tilt around x-axis
    vertical_thickness=3,
    mode="below",
)
surf_tilted.generate(nx, ny, lx, ly)

# ============================================================================
# Step 3: Validate resolution
# ============================================================================
print("\n--- Resolution checks ---")
validate_resolution(surf_bedrock.surface_model, dz, min_cells=3)
validate_resolution(surf_tilted.surface_model, dz, min_cells=3)

# ============================================================================
# Step 4: Insert surfaces into the domain
# ============================================================================

# Bedrock: axis-aligned → use fast voxelize_surface path
bedrock_indices = height_to_indices(surf_bedrock.surface_model, dz)
bedrock_indices += surf_bedrock.reference_point[2] / dz
geometry = voxelize_surface(
    geometry,
    bedrock_indices,
    material_id=ICE,
    vertical_thickness=8,
    mode="two-sided",
)
print(f"After bedrock: unique IDs = {np.unique(geometry)}")

# Tilted layer: rotated → use insert_surface_rotated
geometry = insert_surface_rotated(
    geometry,
    surf_tilted.surface_model,
    material_id=WATER,
    dx=dx, dy=dy, dz=dz,
    reference_point=surf_tilted.reference_point,
    angle_x=surf_tilted.angle_x,
    angle_y=surf_tilted.angle_y,
    angle_z=surf_tilted.angle_z,
    vertical_thickness=3,
    mode="below",
)
print(f"After tilted layer: unique IDs = {np.unique(geometry)}")

# ============================================================================
# Step 5: Write Fortran binary
# ============================================================================
write_geometry(geometry, filename="geometry.dat")
print("\nWrote geometry.dat")

# ============================================================================
# Step 6: Visualise
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# x-z slice at mid-y
axes[0].imshow(
    geometry[:, ny // 2, :].T,
    origin="lower", aspect="auto",
    extent=[0, lx, 0, lz],
    cmap="tab10",
)
axes[0].set_title(f"x–z slice (y = {ny // 2 * dy:.1f} m)")
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("z (m)")

# y-z slice at mid-x
axes[1].imshow(
    geometry[nx // 2, :, :].T,
    origin="lower", aspect="auto",
    extent=[0, ly, 0, lz],
    cmap="tab10",
)
axes[1].set_title(f"y–z slice (x = {nx // 2 * dx:.1f} m)")
axes[1].set_xlabel("y (m)")
axes[1].set_ylabel("z (m)")

# x-y slice at mid-z
axes[2].imshow(
    geometry[:, :, nz // 2].T,
    origin="lower", aspect="auto",
    extent=[0, lx, 0, ly],
    cmap="tab10",
)
axes[2].set_title(f"x–y slice (z = {nz // 2 * dz:.1f} m)")
axes[2].set_xlabel("x (m)")
axes[2].set_ylabel("y (m)")

plt.tight_layout()
plt.savefig("domain_slices.png", dpi=150)
plt.show()
print("Saved domain_slices.png")
