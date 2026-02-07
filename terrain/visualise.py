import numpy as np
import open3d as o3d
from .marching import create_marching_cubes_mesh
from .perlin import generate_noise_grid

def visualize_data(points, values):
    """
    Presentation: Converts raw data into an Open3D PointCloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Normalize values for color mapping (0 to 1)
    v_min, v_max = values.min(), values.max()
    norm_values = (values - v_min) / (v_max - v_min)

    # Color logic: Green for high density, Blue for low density
    colors = np.zeros((len(points), 3))
    colors[:, 1] = norm_values      # Green channel
    colors[:, 2] = 1 - norm_values  # Blue channel

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Add a coordinate frame for orientation reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    print("Opening Open3D Visualizer...")
    o3d.visualization.draw_geometries([pcd, coord_frame])


def visualize_mesh(mesh):
    """
    Opens the Open3D viewer for the generated mesh.
    """
    # Create a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

    print("Opening 3D Mesh Visualizer...")
    # Press 'W' on your keyboard during visualization to see the wireframe!
    o3d.visualization.draw_geometries([mesh, coord_frame],
                                      mesh_show_back_face=True)



class PerlinExplorer:
    def __init__(self, x, y, z, rng, radius, scale=0.1, level=0.1):
        # Initial State
        self.coords = {"x": x, "y": y, "z": z}
        self.rng = rng
        self.radius = radius
        self.scale = scale
        self.level = level

        # Initialize Visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Perlin 3D Explorer", width=1280, height=720)

        # Create Initial Mesh using your existing logic
        self.mesh = self._update_mesh_logic()
        self.vis.add_geometry(self.mesh)

        # Bind Keys (A/S: X, Q/W: Y, Z/X: Z)
        self._bind_keys()
        print(f"Explorer Started at: {self.coords}")
        print("Controls: A/S (X-axis), Q/W (Y-axis), Z/X (Z-axis)")

    def _update_mesh_logic(self):
        """Uses your existing functions to build the mesh."""
        # 1. Use your existing noise generator
        volume = generate_noise_grid(self.coords["x"], self.coords["y"], self.coords["z"],
                                    self.rng, self.radius, self.scale)

        # 2. Use your existing mesh converter
        mesh = create_marching_cubes_mesh(volume, self.level)

        # 3. Apply world offset so the mesh moves with you
        offset = np.array([self.coords["x"], self.coords["y"], self.coords["z"]]) - self.radius
        mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices) + offset)
        return mesh

    def refresh(self, vis):
        """Regenerates the terrain and updates the window."""
        new_mesh = self._update_mesh_logic()

        # Update geometry without resetting the camera view
        self.mesh.vertices = new_mesh.vertices
        self.mesh.triangles = new_mesh.triangles
        self.mesh.compute_vertex_normals()

        self.vis.update_geometry(self.mesh)
        self.vis.update_renderer()
        print(f"Moved to: {self.coords}", end="\r")

    def _bind_keys(self):
        # Key Constants: A=65, S=83, Q=81, W=87, Z=90, X=88
        def move(axis, delta):
            def callback(vis):
                self.coords[axis] += delta
                self.refresh(vis)
            return callback

        self.vis.register_key_callback(65, move("x", 1))   # A
        self.vis.register_key_callback(83, move("x", -1))  # S
        self.vis.register_key_callback(81, move("y", 1))   # Q
        self.vis.register_key_callback(87, move("y", -1))  # W
        self.vis.register_key_callback(90, move("z", 1))   # Z
        self.vis.register_key_callback(88, move("z", -1))  # X

    def start(self):
        self.vis.run()
        self.vis.destroy_window()


'''
explorer = PerlinExplorer(x=0, y=0, z=0, rng=42, radius=15)
explorer.start()

'''
