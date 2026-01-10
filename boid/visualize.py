import torch
import open3d as o3d
import numpy as np
from var import WIDTH, HEIGHT, DEPTH, DT, STEPS
from simulate import step

def create_bounding_box(xmin=0, ymin=0, zmin=0, xmax=WIDTH, ymax=HEIGHT, zmax=DEPTH):
    """
    Creates a 3D bounding box as an Open3D LineSet.

    The bounding box is defined by its minimum and maximum coordinates along the X, Y, and Z axes. It is represented as a wireframe cube with red edges.

    Args:
        xmin (float, optional): Minimum x-coordinate of the box. Defaults to 0.
        ymin (float, optional): Minimum y-coordinate of the box. Defaults to 0.
        zmin (float, optional): Minimum z-coordinate of the box. Defaults to 0.
        xmax (float, optional): Maximum x-coordinate of the box. Defaults to WIDTH.
        ymax (float, optional): Maximum y-coordinate of the box. Defaults to HEIGHT.
        zmax (float, optional): Maximum z-coordinate of the box. Defaults to DEPTH.

    Returns:
        o3d.geometry.LineSet: An Open3D LineSet object representing the bounding box, with vertices at the specified coordinates and red-colored edges.
    """
    points = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top face
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])
    return line_set


def tensor_to_pointcloud(pos_tensor):
    """
    Converts a PyTorch tensor of boid positions into an Open3D PointCloud object.

    This function detaches the tensor from the computation graph, moves it to CPU,converts it to a NumPy array, and then wraps it into an Open3D PointCloud for visualization. All points are painted with a uniform green color.

    Args:
        pos_tensor (torch.Tensor): Tensor of shape (N, 3) representing boid positions.

    Returns:
        o3d.geometry.PointCloud: An Open3D PointCloud object containing the boid positions, with all points colored green.
    """
    pos = pos_tensor.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.paint_uniform_color([0, 1, 0])
    return pcd


def run_simulation(pos, vel, steps=STEPS):
    """
    Runs the boid simulation with Open3D visualization.

    This function initializes an Open3D visualizer window, creates a bounding box to represent the simulation space, and converts boid positions into a point cloud for visualization. It then iteratively updates boid positions and velocities using the `step` function, refreshing the visualization at each iteration. The simulation runs for the specified number of steps and closes the window when finished.

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing initial boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing initial boid velocities.
        steps (int, optional): Number of simulation steps to run. Defaults to STEPS.

    Returns:
        None
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    bbox = create_bounding_box()
    vis.add_geometry(bbox)
    pcd = tensor_to_pointcloud(pos)
    vis.add_geometry(pcd)

    for _ in range(steps):
        pos, vel = step(pos, vel, dt=DT)
        pcd.points = o3d.utility.Vector3dVector(pos.detach().cpu().numpy())
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()
