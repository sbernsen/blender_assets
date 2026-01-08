import bpy
import bmesh
import numpy as np
from mathutils import Vector, bvhtree

class VolumePointSample:
    def __init__(self, obj, layer_index):
        self.obj = obj
        self.layer = layer_index.get(obj.name, 0)
        
        # Local bmesh and BVH
        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        self.bvh = bvhtree.BVHTree.FromBMesh(bm)
        
        # Simple material color (per object)
        self.base_color = self._get_base_color()
    
    def _get_base_color(self):
        """Use the material's Viewport Display color (Material -> Viewport Display)."""
        if not self.obj.material_slots:
            return (1.0, 1.0, 1.0)
        
        mat = self.obj.material_slots[0].material
        if mat is None:
            return (1.0, 1.0, 1.0)
        
        # This is the Viewport Display color (RGBA)
        rgba = list(mat.diffuse_color)  # in 4.x this is the viewport color for
        col = rgba[:3]
        # col = mat.diffuse_color  # float[4]
        return (col[0], col[1], col[2])
    
    def sample_at_world_point(self, p_world):
        """Return (hit, distance, color) for this object at p_world."""
        # Transform point into object local space
        p_local = self.obj.matrix_world.inverted() @ p_world
        hit = self.bvh.find_nearest(p_local)
        if hit is None:
            return False, None, None
        loc, normal, index, dist = hit
        return True, dist, self.base_color



# ------------------------------------------------------------------------------

# List of object names that form your hierarchical stack
OBJECT_NAMES = ["air", "bed", "ice", "layer"]  # ordered how you like

# Optional: a layer index per object (higher wins on tie)
LAYER_INDEX = {
    "ice": 0,
    "air": 1,
    "bed": 2,
    "layer": 3
}

# Volume to sample (world space)
origin = Vector((-25.0, -20.0, 0.0))   # min corner
dims   = Vector((50.0, 40.0, 60.0))   # size
step   = Vector((0.25, 0.25, 0.25))   # spacing


# Collect sample targets from your objects
targets = []
for name in OBJECT_NAMES:
    obj = bpy.data.objects.get(name)
    if obj is None:
        print("Warning: object not found:", name)
        continue
    targets.append(VolumePointSample(obj, LAYER_INDEX))

if not targets:
    raise RuntimeError("No valid objects found to sample from.")

# -----------------------------
# GRID SIZE
# -----------------------------

Nx = int(dims.x / step.x) + 1
Ny = int(dims.y / step.y) + 1
Nz = int(dims.z / step.z) + 1

colors = np.zeros((Nx, Ny, Nz, 3), dtype=float)

# -----------------------------
# MAIN SAMPLING LOOP
# -----------------------------
ind = 0
for ix in range(Nx):
    x = origin.x + ix * step.x
    for iy in range(Ny):
        y = origin.y + iy * step.y
        for iz in range(Nz):
            print(ind)
            ind += 1
            z = origin.z + iz * step.z
            p = Vector((x, y, z))
            best_hit = False
            best_dist = None
            best_layer = None
            best_color = (0.0, 0.0, 0.0)
            # Check all objects at this point
            for t in targets:
                hit, dist, col = t.sample_at_world_point(p)
                if not hit:
                    continue
                if not best_hit:
                    best_hit = True
                    best_dist = dist
                    best_layer = t.layer
                    best_color = col
                else:
                    # Priority: higher layer; tie-breaker: smaller distance
                    if t.layer > best_layer:
                        best_layer = t.layer
                        best_dist = dist
                        best_color = col
                    elif t.layer == best_layer and dist < best_dist:
                        best_dist = dist
                        best_color = col
            colors[ix, iy, iz, :] = best_color

