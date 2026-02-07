import open3d as o3d
from skimage import measure

def create_marching_cubes_mesh(noise_grid, level=0.0):
    """
    Converts 3D noise density into a mesh using Marching Cubes.
    level: The threshold (ISO value) where the surface is drawn.
    """
    # 1. Generate mesh data
    # spacing represents the distance between voxels (1 unit)
    verts, faces, normals, values = measure.marching_cubes(noise_grid, level=level)

    # 2. Convert to Open3D Mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # 3. Finalize mesh properties
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.7, 1.0]) # Light blue "terrain" color

    return mesh
