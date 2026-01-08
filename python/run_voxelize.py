from seidart.routines.classes import VolumeBuilder
from seidart.routines.prjbuild import * 


geometry_priority = {
    "ice": 0,
    "air": 1,
    "base": 2,
    "heterogeneity": 3,
}


labeler = VolumeBuilder(
    obj_path="heterogeneity.obj",
    priority=geometry_priority,
    x_min=-25.0, x_max=25.0, dx=0.25,
    y_min=0.0,   y_max=60.0, dy=0.25,   # OBJ Y = Blender Z
    z_min=-20.0, z_max=20.0, dz=0.25,   # OBJ Z = Blender Y
)

grid = labeler.label_domain()
print("Unique labels:", np.unique(grid))

labeler.plot_slice(102, 'xy')