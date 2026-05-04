"""
Surface roughness generation and 3D FDTD domain construction.

This module generates stochastic rough surfaces (Gaussian or spectral) and
inserts them into a regular 3-D Cartesian grid that is compatible with the
SeidarT CPML-FDTD solver.  Surfaces are 2-D height fields that can be rotated
into any orientation within the domain via three Euler angles and placed
relative to a user-defined reference point.

Array convention
----------------
All 3-D arrays follow the SeidarT / Fortran convention: **(nx, ny, nz)**.
The x-axis is the first index, y is the second, and z (depth) is the third.

Staircase approximation
-----------------------
A rough surface discretised onto a Cartesian grid becomes a staircase.  This
is accurate when the dominant wavelength of the propagating wave is much
larger than the grid spacing.  For high-frequency simulations where
λ ≈ roughness correlation length, the staircase introduces spurious numerical
scattering.  Ensure that the grid spacing satisfies the wavenumber band-limit
criterion (dx, dy, dz ≤ λ_min / 4) *and* that the RMS roughness amplitude
spans at least 2–3 grid cells in the insertion direction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import FortranFile
from scipy.spatial.transform import Rotation

import gstools as gs
from gstools.field.generator import Fourier


# =============================================================================
# ============================== Surface Fields ===============================
# =============================================================================

def gaussian_field(
        variance: float,
        length_scale,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        *,
        dim: int = 2,
        angles: float = 0.0,
        seed: int = 42,
    ) -> np.ndarray:
    """Generate a 2-D Gaussian random-field height map.

    Parameters
    ----------
    variance : float
        Variance (σ²) of the surface heights.
    length_scale : float or list of float
        Correlation length(s).  A scalar gives an isotropic field; a
        two-element list ``[lx, ly]`` gives anisotropy.
    nx, ny : int
        Number of grid points along the x- and y-directions.
    lx, ly : float
        Physical domain lengths (metres) along x and y.
    dim : int, optional
        Dimensionality of the covariance model (always 2 for a surface).
    angles : float, optional
        Rotation angle (degrees) of the anisotropy ellipse.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    z : ndarray, shape (nx, ny)
        Height values in physical units (metres).  To convert to grid-cell
        indices, divide by the grid spacing in the insertion direction.
    """
    model = gs.Gaussian(
        dim=dim,
        var=variance,
        len_scale=length_scale,
        angles=angles,
    )
    srf = gs.SRF(model, seed=seed)
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    z = srf(
        (X.ravel(), Y.ravel()), mesh_type="unstructured"
    ).reshape(Y.shape)
    return z.T  # transpose from (ny, nx) to (nx, ny)


def spectral_field(
        variance: float,
        length_scale,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        *,
        dim: int = 2,
        mode_no: int = 512,
        seed: int = 42,
    ) -> np.ndarray:
    """Generate a 2-D surface using a Fourier spectral generator.

    Parameters
    ----------
    variance : float
        Variance (σ²) of the surface heights.
    length_scale : float or list of float
        Correlation length(s).
    nx, ny : int
        Number of grid points along x and y.
    lx, ly : float
        Physical domain lengths (metres) along x and y.
    dim : int, optional
        Dimensionality of the covariance model (always 2).
    mode_no : int, optional
        Number of Fourier modes per axis.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    z : ndarray, shape (nx, ny)
        Height values in physical units (metres).
    """
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)

    model = gs.Gaussian(dim=dim, var=variance, len_scale=length_scale)
    srf = gs.SRF(
        model,
        generator="Fourier",
        period=[np.ptp(x), np.ptp(y)],
        mode_no=[mode_no, mode_no],
        seed=seed,
    )
    z = srf.structured((x, y))
    return z.T  # transpose from (ny, nx) to (nx, ny)


# =============================================================================
# =============================== Utilities ===================================
# =============================================================================

def height_to_indices(
        height_field: np.ndarray,
        grid_spacing: float,
    ) -> np.ndarray:
    """Convert a physical-units height field to grid-cell indices.

    Parameters
    ----------
    height_field : ndarray
        Surface heights in metres.
    grid_spacing : float
        Grid spacing (dx, dy, or dz) in metres along the insertion direction.

    Returns
    -------
    index_field : ndarray (same shape as *height_field*)
        Surface heights expressed as (fractional) grid-cell indices.
    """
    return height_field / grid_spacing


def validate_resolution(
        height_field: np.ndarray,
        grid_spacing: float,
        min_cells: int = 3,
    ) -> bool:
    """Check whether the roughness amplitude is resolvable on the grid.

    The RMS amplitude of the surface must span at least *min_cells* grid
    cells for the staircase approximation to be meaningful.

    Parameters
    ----------
    height_field : ndarray
        Surface heights in metres.
    grid_spacing : float
        Grid spacing in the insertion direction (metres).
    min_cells : int, optional
        Minimum number of cells the RMS roughness should span.

    Returns
    -------
    ok : bool
        ``True`` if the resolution criterion is satisfied.

    Raises
    ------
    UserWarning
        Printed (not raised) when the criterion is not met.
    """
    rms = np.std(height_field)
    n_cells = rms / grid_spacing
    if n_cells < min_cells:
        print(
            f"WARNING: RMS roughness = {rms:.4g} m spans only "
            f"{n_cells:.1f} cells (need >= {min_cells}).  "
            f"Reduce grid_spacing or increase surface variance."
        )
        return False
    return True


# =============================================================================
# ======================== 3-D Domain Construction ============================
# =============================================================================

def extrude_domain_3d(
        geometry_array: np.ndarray,
        ny: int,
    ) -> np.ndarray:
    """Extrude a 2-D cross-section into a uniform 3-D domain.

    Parameters
    ----------
    geometry_array : ndarray, shape (nx, nz)
        Integer material-ID array in the x–z plane.
    ny : int
        Number of grid points in the y-direction.

    Returns
    -------
    labels : ndarray, shape (nx, ny, nz)
        3-D integer geometry array (SeidarT convention).
    """
    geometry_array = np.asarray(geometry_array)
    if geometry_array.ndim != 2:
        raise ValueError("Geometry must be a 2-D array of shape (nx, nz).")

    nx, nz = geometry_array.shape
    # Broadcast along the new y-axis (axis 1)
    labels = np.tile(geometry_array[:, None, :], (1, ny, 1))  # (nx, ny, nz)
    return labels


def _build_rotation_matrix(
        angle_x: float,
        angle_y: float,
        angle_z: float,
    ) -> np.ndarray:
    """Build a 3×3 rotation matrix from three Euler angles (degrees).

    Rotation order is intrinsic Z-Y-X (i.e. rotate around z first, then y,
    then x).

    Parameters
    ----------
    angle_x, angle_y, angle_z : float
        Rotation angles in degrees around the x-, y-, and z-axes respectively.

    Returns
    -------
    R : ndarray, shape (3, 3)
    """
    r = Rotation.from_euler("ZYX", [angle_z, angle_y, angle_x], degrees=True)
    return r.as_matrix()


def voxelize_surface(
        geometry_3d: np.ndarray,
        surface_model: np.ndarray,
        *,
        material_id: int,
        vertical_thickness=1,
        vertical_shift: int = 0,
        mode: str = "below",
        clamp: bool = True,
    ) -> np.ndarray:
    """Stamp a 2-D surface into a 3-D geometry array along the z-axis.

    The *surface_model* gives the z-index of the surface at each (x, y)
    location.  A band of voxels of width *vertical_thickness* is assigned the
    given *material_id*.

    Parameters
    ----------
    geometry_3d : ndarray, shape (nx, ny, nz)
        Mutable integer geometry array.
    surface_model : ndarray, shape (nx, ny)
        Surface height expressed as **z-index** values.
    material_id : int
        Integer label to write into the affected voxels.
    vertical_thickness : int or ndarray of int, shape (nx, ny)
        Number of z-cells to fill.  A scalar applies uniformly; a 2-D array
        allows spatially variable thickness.
    vertical_shift : int, optional
        Constant offset (in z-cells) applied to *surface_model* before
        thickness is computed.
    mode : {'below', 'above', 'two-sided'}
        Where to place the band relative to the surface centre.
    clamp : bool, optional
        If ``True``, indices are clamped to ``[0, nz)``; otherwise an
        ``IndexError`` is raised when the layer exceeds the domain.

    Returns
    -------
    out : ndarray, shape (nx, ny, nz)
        Copy of *geometry_3d* with the surface layer written.
    """
    if geometry_3d.ndim != 3:
        raise ValueError(
            "geometry_3d must be a 3-D array shaped (nx, ny, nz)."
        )
    nx, ny, nz = geometry_3d.shape

    surface_model = np.asarray(surface_model)
    if surface_model.shape != (nx, ny):
        raise ValueError(
            f"surface_model must have shape (nx, ny)=({nx}, {ny}), "
            f"but got {surface_model.shape}."
        )

    # Support scalar or 2-D thickness
    vertical_thickness = np.broadcast_to(
        np.asarray(vertical_thickness, dtype=np.int64),
        (nx, ny),
    ).copy()
    if (vertical_thickness < 1).any():
        raise ValueError("vertical_thickness must be >= 1 everywhere.")

    # Centre indices at each (x, y)
    k_center = np.rint(surface_model).astype(np.int64) + int(vertical_shift)

    T = vertical_thickness
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
        k0 = np.clip(k0, 0, nz)
        k1 = np.clip(k1, 0, nz)
    else:
        if (k0 < 0).any() or (k1 > nz).any():
            raise IndexError("Surface layer extends outside z bounds.")

    # Vectorised boolean mask
    Z = np.arange(nz, dtype=np.int64)[None, None, :]        # (1, 1, nz)
    start = k0[:, :, None]                                   # (nx, ny, 1)
    stop = k1[:, :, None]                                    # (nx, ny, 1)
    layer_mask = (Z >= start) & (Z < stop)                   # (nx, ny, nz)

    out = np.array(geometry_3d, copy=True)
    out[layer_mask] = int(material_id)
    return out


def insert_surface_rotated(
        geometry_3d: np.ndarray,
        surface_model: np.ndarray,
        *,
        material_id: int,
        dx: float,
        dy: float,
        dz: float,
        reference_point: tuple = (0, 0, 0),
        angle_x: float = 0.0,
        angle_y: float = 0.0,
        angle_z: float = 0.0,
        vertical_thickness: int = 1,
        mode: str = "below",
        clamp: bool = True,
    ) -> np.ndarray:
    """Insert a rotated 2-D surface into a 3-D geometry array.

    The surface is generated in its own local frame, then rotated by three
    Euler angles and translated so that its origin aligns with
    *reference_point* in the domain.  The reference point ``(0, 0, 0)``
    corresponds to the domain origin **before** CPML padding is added
    (i.e. grid-index ``(cpml, cpml, cpml)`` in the padded array).

    The surface can be larger or smaller than the domain—only the portion
    that overlaps the grid is stamped.

    Parameters
    ----------
    geometry_3d : ndarray, shape (nx, ny, nz)
        Mutable integer geometry array.
    surface_model : ndarray, shape (ns_x, ns_y)
        Surface heights in **metres** (physical units), defined on a grid
        whose spacing matches (dx, dy).
    material_id : int
        Material label for the surface layer.
    dx, dy, dz : float
        Grid spacings in metres.
    reference_point : tuple of float, optional
        ``(x0, y0, z0)`` in metres — the point in the domain where the
        surface origin is placed.
    angle_x, angle_y, angle_z : float, optional
        Rotation angles in degrees around the x-, y-, and z-axes.
    vertical_thickness : int, optional
        Layer thickness in grid cells (applied after rotation).
    mode : {'below', 'above', 'two-sided'}
        Placement relative to the surface centre.
    clamp : bool, optional
        Clamp indices to domain bounds.

    Returns
    -------
    out : ndarray, shape (nx, ny, nz)
        Updated geometry with the rotated surface inserted.
    """
    if geometry_3d.ndim != 3:
        raise ValueError(
            "geometry_3d must be a 3-D array shaped (nx, ny, nz)."
        )
    nx, ny, nz = geometry_3d.shape
    ns_x, ns_y = surface_model.shape

    # Build local coordinates of each surface point
    R = _build_rotation_matrix(angle_x, angle_y, angle_z)
    x0, y0, z0 = reference_point

    out = np.array(geometry_3d, copy=True)

    for si in range(ns_x):
        for sj in range(ns_y):
            # Local position: surface lives in the x-y plane with height in z
            local = np.array([si * dx, sj * dy, surface_model[si, sj]])
            # Rotate and translate
            world = R @ local + np.array([x0, y0, z0])
            # Convert to grid indices
            ix = int(np.round(world[0] / dx))
            iy = int(np.round(world[1] / dy))
            iz_center = int(np.round(world[2] / dz))

            # Apply thickness
            T = int(vertical_thickness)
            if mode == "below":
                iz0, iz1 = iz_center, iz_center + T
            elif mode == "above":
                iz0, iz1 = iz_center - (T - 1), iz_center + 1
            elif mode == "two-sided":
                iz0 = iz_center - T // 2
                iz1 = iz_center + (T - T // 2)
            else:
                raise ValueError(
                    "mode must be 'below', 'above', or 'two-sided'."
                )

            if clamp:
                iz0, iz1 = max(iz0, 0), min(iz1, nz)
            else:
                if iz0 < 0 or iz1 > nz:
                    raise IndexError(
                        "Surface layer extends outside z bounds."
                    )

            # Only stamp if inside domain bounds
            if 0 <= ix < nx and 0 <= iy < ny and iz0 < iz1:
                out[ix, iy, iz0:iz1] = int(material_id)

    return out


# =============================================================================
# ============================ Fortran I/O ====================================
# =============================================================================

def write_geometry(
        geometry_3d: np.ndarray,
        filename: str = "geometry.dat",
    ) -> None:
    """Write a 3-D integer geometry array in SeidarT-compatible Fortran binary.

    The array is written in Fortran (column-major) order using
    ``scipy.io.FortranFile`` so the Fortran solver can read it directly with
    ``read_geometry``.

    Parameters
    ----------
    geometry_3d : ndarray, shape (nx, ny, nz)
        Integer material-ID array.
    filename : str, optional
        Output filename.
    """
    f = FortranFile(filename, "w")
    f.write_record(np.asfortranarray(geometry_3d).astype(np.int32))
    f.close()


# =============================================================================
# ====================== SeidarT Project Integration ==========================
# =============================================================================

def build_seidart_surfaces(
        project_json: str,
        surfaces: list,
        geometry_3d: np.ndarray,
    ) -> np.ndarray:
    """Insert multiple rough surfaces into a domain and update the project JSON.

    Each surface is a dictionary describing one rough interface.  After all
    surfaces are stamped, the geometry is written to ``geometry.dat`` and the
    project JSON is updated with the new material entries.

    Parameters
    ----------
    project_json : str
        Path to the SeidarT project JSON file.
    surfaces : list of dict
        Each dict must contain at minimum::

            {
                'name': str,          # material name, e.g. 'rough_ice'
                'rgb':  str,          # 'R/G/B' colour string
                'id':   int,          # unique material ID
                'surface_model': ndarray,  # 2-D height field (metres)
                'dx': float,          # grid spacings
                'dy': float,
                'dz': float,
                'vertical_thickness': int,
                # Optional rotation / placement
                'reference_point': (x, y, z),  # metres, default (0,0,0)
                'angle_x': float,              # degrees, default 0
                'angle_y': float,
                'angle_z': float,
                'mode': str,                   # 'below'/'above'/'two-sided'
            }

    geometry_3d : ndarray, shape (nx, ny, nz)
        Base geometry array (modified in place through successive stamps).

    Returns
    -------
    geometry_3d : ndarray, shape (nx, ny, nz)
        Updated geometry with all surfaces inserted.

    Notes
    -----
    After calling this function, use the standard SeidarT pipeline
    (``loadproject`` → ``Model.build``) to compute CPML, tensor ``.dat``
    files, and run the solver.
    """
    import json

    # Read existing project
    with open(project_json, "r") as fh:
        data = json.load(fh)

    for surf in surfaces:
        # Stamp surface into geometry
        ref = surf.get("reference_point", (0, 0, 0))
        ax = surf.get("angle_x", 0.0)
        ay = surf.get("angle_y", 0.0)
        az = surf.get("angle_z", 0.0)
        md = surf.get("mode", "below")

        has_rotation = (ax != 0.0 or ay != 0.0 or az != 0.0)

        if has_rotation:
            geometry_3d = insert_surface_rotated(
                geometry_3d,
                surf["surface_model"],
                material_id=surf["id"],
                dx=surf["dx"],
                dy=surf["dy"],
                dz=surf["dz"],
                reference_point=ref,
                angle_x=ax,
                angle_y=ay,
                angle_z=az,
                vertical_thickness=surf.get("vertical_thickness", 1),
                mode=md,
            )
        else:
            surface_indices = height_to_indices(
                surf["surface_model"], surf["dz"]
            )
            # Offset to reference z-index
            surface_indices += ref[2] / surf["dz"]
            geometry_3d = voxelize_surface(
                geometry_3d,
                surface_indices,
                material_id=surf["id"],
                vertical_thickness=surf.get("vertical_thickness", 1),
                mode=md,
            )

        # Append material entry to the project JSON
        mat_entry = {
            "id": surf["id"],
            "name": surf["name"],
            "rgb": surf["rgb"],
            "temperature": surf.get("temperature", None),
            "density": surf.get("density", None),
            "porosity": surf.get("porosity", None),
            "water_content": surf.get("water_content", None),
            "is_anisotropic": surf.get("is_anisotropic", None),
            "euler_angles": surf.get("euler_angles", None),
        }
        # Check if id already exists; if so, update; otherwise append
        existing_ids = [m["id"] for m in data["Materials"]]
        if surf["id"] in existing_ids:
            idx = existing_ids.index(surf["id"])
            data["Materials"][idx] = mat_entry
        else:
            data["Materials"].append(mat_entry)

        # Update nmats
        data["Domain"]["nmats"] = len(data["Materials"])

    # Write updated JSON
    with open(project_json, "w") as fh:
        json.dump(data, fh, indent=4, default=_numpy_encoder)

    # Write geometry binary
    write_geometry(geometry_3d)

    return geometry_3d


def _numpy_encoder(obj):
    """JSON encoder fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
