"""surface_roughness — stochastic surface generation for FDTD domains."""

from surface_roughness.classes.classes import RoughSurface
from surface_roughness.classes.definitions import (
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

__all__ = [
    "RoughSurface",
    "gaussian_field",
    "spectral_field",
    "height_to_indices",
    "validate_resolution",
    "extrude_domain_3d",
    "voxelize_surface",
    "insert_surface_rotated",
    "write_geometry",
    "build_seidart_surfaces",
]
