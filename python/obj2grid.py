import numpy as np
import trimesh

OBJ_PATH = "heterogeneity.obj"
scene = trimesh.load(OBJ_PATH, process=False)
geoms = scene.geometry if isinstance(scene, trimesh.Scene) else {"mesh": scene}

# -------------------------
# PRIORITY AND LABEL SETUP
# -------------------------

# Higher number = higher priority (later in stack, overwrites others)
priority = {
    "ice": 0,
    "air": 1,
    "base": 2,
    "heterogeneity": 3,  # all heterogeneity* share this priority
}

# Map objects to labels (material IDs)
label_for_name = {}
for name in geoms.keys():
    lower = name.lower()
    if "ice" in lower:
        label_for_name[name] = 2
    elif "base" in lower:
        label_for_name[name] = 3
    elif "air" in lower:
        label_for_name[name] = 4
    elif "heterogeneity" in lower:
        label_for_name[name] = 10   # common label for all heterogeneities
    else:
        label_for_name[name] = 99   # fallback

# Helper: get priority key for sorting
def get_priority_key(name: str) -> int:
    lower = name.lower()
    if "heterogeneity" in lower:
        return priority["heterogeneity"]
    for key in ("base", "air", "ice"):
        if key in lower:
            return priority[key]
    return -1  # lowest

# -------------------------
# BUILD GRID (assumed)
# -------------------------
# xs, ys, zs, points already built as before
# labels = np.full(points.shape[0], BACKGROUND_LABEL, dtype=np.int32)

# -------------------------
# LABEL IN PRIORITY ORDER
# -------------------------

names_sorted = sorted(geoms.keys(), key=get_priority_key)

for name in names_sorted:
    mesh = geoms[name]
    lower = name.lower()
    label = label_for_name[name]
    p = get_priority_key(name)
    
    print(f"Processing '{name}' label={label} priority={p}")
    
    bounds_min, bounds_max = mesh.bounds
    
    coarse = (
        (points[:, 0] >= bounds_min[0]) & (points[:, 0] <= bounds_max[0]) &
        (points[:, 1] >= bounds_min[1]) & (points[:, 1] <= bounds_max[1]) &
        (points[:, 2] >= bounds_min[2]) & (points[:, 2] <= bounds_max[2])
    )
    
    candidate_idx = np.where(coarse)[0]
    if candidate_idx.size == 0:
        print("  No points in bounding box; skipping.")
        continue
    
    pts_candidate = points[candidate_idx]
    
    # For heterogeneities (possibly rotated), refine with contains
    if "heterogeneity" in lower:
        inside_local = mesh.contains(pts_candidate)
        labels[candidate_idx[inside_local]] = label
        print(f"  Heterogeneity: coarse {candidate_idx.size}, inside {inside_local.sum()}")
    else:
        # For axis-aligned cubes (ice, air, base), AABB is enough
        labels[candidate_idx] = label
        print(f"  Bulk region: labeled {candidate_idx.size} cells (AABB)")



# After labeling is complete and you have nx, ny, nz and 1D labels
label_grid = labels.reshape((nx, ny, nz))

print("label_grid shape:", label_grid.shape)  # should be (nx, ny, nz)

def show_slice(arr, index, plane="xy"):
    Nx, Ny, Nz = arr.shape
    if plane == "xy":
        assert 0 <= index < Nz
        img = arr[:, :, index]
        xlabel, ylabel = "ix", "iy"
        title = f"Slice plane=xy, iz={index}"
    elif plane == "xz":
        assert 0 <= index < Ny
        img = arr[:, index, :]
        xlabel, ylabel = "ix", "iz"
        title = f"Slice plane=xz, iy={index}"
    elif plane == "yz":
        assert 0 <= index < Nx
        img = arr[index, :, :]
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

show_slice(label_grid, 80, plane = 'xz')