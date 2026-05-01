"""
Surface roughness class interface.

Provides a ``RoughSurface`` class that wraps surface generation, domain
insertion, and SeidarT project-file bookkeeping into a single object.
"""

import numpy as np
from .definitions import (
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


class RoughSurface:
    """Descriptor for one rough surface layer.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``'rough_bedrock'``).
    material_id : int
        Unique integer ID used in the geometry array and SeidarT project file.
    rgb : str
        Colour string in ``'R/G/B'`` format for project-file bookkeeping.
    variance : float
        Variance (σ²) of the surface heights.
    length_scale : float or list of float
        Correlation length(s) for the random field.
    method : {'gaussian', 'spectral'}
        Surface generation method.
    angles : float, optional
        Anisotropy angle (degrees) for the random-field generator.
    seed : int, optional
        Random seed for reproducibility.
    reference_point : tuple of float, optional
        ``(x, y, z)`` in metres — where the surface origin maps to in the
        domain.  ``(0, 0, 0)`` aligns with the domain origin *before* CPML
        padding.
    angle_x, angle_y, angle_z : float, optional
        Euler rotation angles (degrees) for orienting the surface in 3-D.
    vertical_thickness : int, optional
        Layer thickness in grid cells.
    mode : {'below', 'above', 'two-sided'}, optional
        How the thickness band is placed relative to the surface.
    """

    def __init__(
        self,
        name: str,
        material_id: int,
        rgb: str,
        variance: float,
        length_scale,
        *,
        method: str = "gaussian",
        angles: float = 0.0,
        seed: int = 42,
        reference_point: tuple = (0.0, 0.0, 0.0),
        angle_x: float = 0.0,
        angle_y: float = 0.0,
        angle_z: float = 0.0,
        vertical_thickness: int = 1,
        mode: str = "below",
    ):
        self.name = name
        self.material_id = material_id
        self.rgb = rgb
        self.variance = variance
        self.length_scale = length_scale
        self.method = method.lower()
        self.angles = angles
        self.seed = seed
        self.reference_point = reference_point
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.vertical_thickness = vertical_thickness
        self.mode = mode
        self.surface_model = None  # populated by generate()

    def generate(self, nx, ny, lx, ly):
        """Generate the surface height field.

        Parameters
        ----------
        nx, ny : int
            Number of grid points along x and y.
        lx, ly : float
            Physical domain lengths (metres).

        Returns
        -------
        surface_model : ndarray, shape (ny, nx)
            Height values in metres.
        """
        if self.method == "gaussian":
            self.surface_model = gaussian_field(
                self.variance,
                self.length_scale,
                nx, ny, lx, ly,
                angles=self.angles,
                seed=self.seed,
            )
        elif self.method == "spectral":
            self.surface_model = spectral_field(
                self.variance,
                self.length_scale,
                nx, ny, lx, ly,
                seed=self.seed,
            )
        else:
            raise ValueError(
                f"Unknown method '{self.method}'. Use 'gaussian' or 'spectral'."
            )
        return self.surface_model

    def to_dict(self, dx, dy, dz):
        """Export as a dict consumable by ``build_seidart_surfaces``.

        Parameters
        ----------
        dx, dy, dz : float
            Grid spacings in metres.

        Returns
        -------
        d : dict
        """
        if self.surface_model is None:
            raise RuntimeError("Call generate() before to_dict().")
        return {
            "name": self.name,
            "id": self.material_id,
            "rgb": self.rgb,
            "surface_model": self.surface_model,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "reference_point": self.reference_point,
            "angle_x": self.angle_x,
            "angle_y": self.angle_y,
            "angle_z": self.angle_z,
            "vertical_thickness": self.vertical_thickness,
            "mode": self.mode,
        }
