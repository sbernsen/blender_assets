import numpy as np
import trimesh
import matplotlib.pyplot as plt


class VolumeBuilder:
    """
    Discretize an OBJ scene onto a regular FD grid with region labels.
    
    Parameters
    ----------
    obj_path : str
        Path to the OBJ file.
    priority : dict
        Dict mapping group names to priority (higher = later overwrite).
        Example: {"ice": 0, "air": 1, "base": 2, "heterogeneity": 3}
    x_min, x_max, dx : float
    y_min, y_max, dy : float
    z_min, z_max, dz : float
        Grid extents and spacings in OBJ coordinates.
    """
    
    def __init__(
            self,
            obj_path,
            priority,
            x_min, x_max, dx,
            y_min, y_max, dy,
            z_min, z_max, dz,
            background_label=0
        ):
        self.obj_path = obj_path
        self.priority = priority
        self.background_label = background_label
        
        # Load scene
        scene = trimesh.load(self.obj_path, process=False)
        if isinstance(scene, trimesh.Scene):
            self.geoms = scene.geometry  # dict: name -> Trimesh [web:70][web:75]
        else:
            self.geoms = {"mesh": scene}
        
        # Build grid
        self.xs = np.arange(x_min, x_max + 0.5 * dx, dx)
        self.ys = np.arange(y_min, y_max + 0.5 * dy, dy)
        self.zs = np.arange(z_min, z_max + 0.5 * dz, dz)
        
        self.nx, self.ny, self.nz = len(self.xs), len(self.ys), len(self.zs)
        
        X, Y, Z = np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")
        self.points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        self.labels_1d = np.full(self.points.shape[0],
                                 self.background_label,
                                 dtype=np.int32)
        self.label_grid = None
        
        # Precompute label map per object name
        self.label_for_name = self._build_label_map()
    
    # -------------------------
    # internal helpers
    # -------------------------
    def _match_tag(self, name: str):
        """
        Find which priority tag applies to this geometry name.
        
        Returns
        -------
        tag : str or None
            The matched tag (key in self.priority) or None if no match.
        """
        lower = name.lower()
        for tag in self.priority.keys():
            if tag in lower:
                return tag
        return None
    
    def _get_priority_for_name(self, name: str) -> int:
        """
        Get the numeric priority for a geometry name.
        """
        tag = self._match_tag(name)
        if tag is None:
            return -1  # lowest
        return self.priority[tag]
    
    def _get_label_for_name(self, name: str) -> int:
        """
        Label ID = priority value associated with matched tag.
        """
        tag = self._match_tag(name)
        if tag is None:
            return self.background_label
        return self.priority[tag]
    
    def _build_label_map(self):
        """
        Map geometry names to integer labels (material IDs).
        """
        label_for_name = {}
        for name in self.geoms.keys():
            lower = name.lower()
            if "ice" in lower:
                label_for_name[name] = 2
            elif "base" in lower:
                label_for_name[name] = 3
            elif "air" in lower:
                label_for_name[name] = 4
            elif "heterogeneity" in lower:
                label_for_name[name] = 10   # common label
            else:
                label_for_name[name] = 99   # fallback
        return label_for_name
    
    # -------------------------
    # main API
    # -------------------------
    
    def label_domain(self):
        """
        Fill self.labels_1d and self.label_grid using priority-based overwrite.
        
        Returns
        -------
        label_grid : np.ndarray, shape (nx, ny, nz)
        """
        names_sorted = sorted(self.geoms.keys(), key=self._get_priority_for_name)
        
        for name in names_sorted:
            mesh = self.geoms[name]
            tag = self._match_tag(name)
            prio = self._get_priority_for_name(name)
            label = self._get_label_for_name(name)
            
            if tag is None:
                print(f"Skipping '{name}' (no priority tag match).")
                continue
            
            print(f"Processing '{name}' tag='{tag}' label={label} priority={prio}")
            
            bounds_min, bounds_max = mesh.bounds
            
            coarse = (
                (self.points[:, 0] >= bounds_min[0]) & (self.points[:, 0] <= bounds_max[0]) &
                (self.points[:, 1] >= bounds_min[1]) & (self.points[:, 1] <= bounds_max[1]) &
                (self.points[:, 2] >= bounds_min[2]) & (self.points[:, 2] <= bounds_max[2])
            )
            
            candidate_idx = np.where(coarse)[0]
            if candidate_idx.size == 0:
                print("  No points in bounding box; skipping.")
                continue
            
            pts_candidate = self.points[candidate_idx]
            
            # Example: treat "heterogeneity" as possibly rotated (needs contains),
            # others as axis-aligned (AABB enough).
            if tag == "heterogeneity":
                inside_local = mesh.contains(pts_candidate)  # [web:82][web:114]
                self.labels_1d[candidate_idx[inside_local]] = label
                print(f"  Heterogeneity: coarse {candidate_idx.size}, inside {inside_local.sum()}")
            else:
                self.labels_1d[candidate_idx] = label
                print(f"  Bulk region: labeled {candidate_idx.size} cells (AABB)")
        
        self.label_grid = self.labels_1d.reshape((self.nx, self.ny, self.nz))
        print("label_grid shape:", self.label_grid.shape)
        return self.label_grid
    
    def _get_priority_key(self, name: str) -> int:
        """
        Return priority for sorting based on the name and self.priority.
        Higher value == processed later.
        """
        lower = name.lower()
        if "heterogeneity" in lower and "heterogeneity" in self.priority:
            return self.priority["heterogeneity"]
        for key in self.priority.keys():
            if key != "heterogeneity" and key in lower:
                return self.priority[key]
        return -1  # lowest priority
    
    # -------------------------
    # visualization helper
    # -------------------------
    
    def show_slice(self, index, plane="xy"):
        """
        Show a 2D slice of a 3D array along one of the principal planes.
        plane: 'xy', 'xz', or 'yz'
        """
        Nx, Ny, Nz = self.label_grid.shape
        if plane == "xy":
            assert 0 <= index < Nz
            img = self.label_grid[:, :, index]
            xlabel, ylabel = "ix", "iy"
            title = f"Slice plane=xy, iz={index}"
        elif plane == "xz":
            assert 0 <= index < Ny
            img = self.label_grid[:, index, :]
            xlabel, ylabel = "ix", "iz"
            title = f"Slice plane=xz, iy={index}"
        elif plane == "yz":
            assert 0 <= index < Nx
            img = self.label_grid[index, :, :]
            xlabel, ylabel = "iy", "iz"
            title = f"Slice plane=yz, ix={index}"
        else:
            raise ValueError("plane must be one of 'xy', 'xz', 'yz'")
        
        plt.figure(figsize=(6, 5))
        plt.imshow(img)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()
