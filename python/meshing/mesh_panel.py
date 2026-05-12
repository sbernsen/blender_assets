import bpy
import os

# --- 1. SETUP CUSTOM PROPERTIES ---
# This adds seismic parameters to every object in your scene
def init_properties():
    bpy.types.Object.porosity = bpy.props.FloatProperty(name="Porosity", default=0.3, min=0.0, max=1.0)
    bpy.types.Object.density = bpy.props.FloatProperty(name="Density (kg/m3)", default=2500.0)
    bpy.types.Object.permeability = bpy.props.FloatProperty(name="Permeability (m2)", default=1e-12)
    bpy.types.Object.is_seismic_domain = bpy.props.BoolProperty(name="Include in Simulation", default=True)

# --- 2. OPERATORS (The Buttons) ---

class SEISMIC_OT_GenerateMesh(bpy.types.Operator):
    """Convert curves/vectors to mesh for inspection"""
    bl_idname = "seismic.generate_mesh"
    bl_label = "Generate Inspection Mesh"

    def execute(self, context):
        # Converts curves to meshes without destroying original
        bpy.ops.object.duplicate()
        bpy.ops.object.convert(target='MESH')
        context.active_object.name = "INSPECTION_MESH"
        return {'FINISHED'}

class SEISMIC_OT_RemoveMesh(bpy.types.Operator):
    """Delete the temporary inspection mesh"""
    bl_idname = "seismic.remove_mesh"
    bl_label = "Remove Inspection Mesh"

    def execute(self, context):
        if "INSPECTION_MESH" in bpy.data.objects:
            obj = bpy.data.objects["INSPECTION_MESH"]
            bpy.data.objects.remove(obj, do_unlink=True)
        return {'FINISHED'}

class SEISMIC_OT_ExportAll(bpy.types.Operator):
    """Export OBJ/MTL and a Data file for SeidarT"""
    bl_idname = "seismic.export_all"
    bl_label = "Export for SeidarT"

    def execute(self, context):
        target_dir = bpy.path.abspath("//") # Saves next to your .blend file
        obj_path = os.path.join(target_dir, "simulation_domain.obj")
        meta_path = os.path.join(target_dir, "material_params.txt")

        # Export OBJ
        bpy.ops.wm.obj_export(filepath=obj_path, export_selected=True)

        # Export Custom Parameters to Text for SeidarT backend
        selected_obj = context.active_object
        with open(meta_path, "w") as f:
            f.write(f"Object: {selected_obj.name}\n")
            f.write(f"Density: {selected_obj.density}\n")
            f.write(f"Porosity: {selected_obj.porosity}\n")
            f.write(f"Permeability: {selected_obj.permeability}\n")
        
        self.report({'INFO'}, f"Exported to {target_dir}")
        return {'FINISHED'}

# --- 3. THE UI PANEL ---

class SEISMIC_PT_Panel(bpy.types.Panel):
    bl_label = "SeidarT Domain Tools"
    bl_idname = "SEISMIC_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Seismic'

    def draw(self, context):
        layout = self.layout
        obj = context.object

        if obj:
            layout.label(text=f"Properties: {obj.name}")
            col = layout.column()
            col.prop(obj, "is_seismic_domain")
            col.prop(obj, "density")
            col.prop(obj, "porosity")
            col.prop(obj, "permeability")
            
            layout.separator()
            layout.operator("seismic.generate_mesh", icon='MESH_DATA')
            layout.operator("seismic.remove_mesh", icon='TRASH')
            layout.separator()
            layout.operator("seismic.export_all", icon='EXPORT')

# --- REGISTRATION ---

def register():
    init_properties()
    bpy.utils.register_class(SEISMIC_OT_GenerateMesh)
    bpy.utils.register_class(SEISMIC_OT_RemoveMesh)
    bpy.utils.register_class(SEISMIC_OT_ExportAll)
    bpy.utils.register_class(SEISMIC_PT_Panel)

def unregister():
    bpy.utils.unregister_class(SEISMIC_OT_GenerateMesh)
    bpy.utils.unregister_class(SEISMIC_OT_RemoveMesh)
    bpy.utils.unregister_class(SEISMIC_OT_ExportAll)
    bpy.utils.unregister_class(SEISMIC_PT_Panel)

if __name__ == "__main__":
    register()