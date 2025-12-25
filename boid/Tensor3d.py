import numpy as np
import open3d as o3d
import torch

"""
Boid Simulation Configuration

Global constants used throughout the boid simulation, including simulation space dimensions, number of boids, device settings, and flocking rule parameters.

Constants:
    WIDTH (int): Width of the 3D simulation space.
    HEIGHT (int): Height of the 3D simulation space.
    DEPTH (int): Depth of the 3D simulation space.

    N (int): Number of boids in the simulation.

    DEVICE (str): Torch device to use ("cpu" or "cuda").

    SPEED (float): Target constant speed for all boids.

    S_RADIUS (float): Minimum separation distance between boids.
    C_RADIUS (float): Perception radius for cohesion (neighbors considered).
    A_RADIUS (float): Perception radius for alignment (neighbors considered).

    S_STRENGTH (float): Strength of separation steering.
    C_STRENGTH (float): Strength of cohesion steering.
    A_STRENGTH (float): Strength of alignment steering.

    DT (float): Time step for position updates.
    STEPS (int): Number of simulation steps to run.
"""

WIDTH = 1000
HEIGHT = 1000
DEPTH = 1000

N = 500

DEVICE = "cuda"

SPEED = 2.0

S_RADIUS = 20.0
C_RADIUS = 50.0
A_RADIUS = 50.0

S_STRENGTH = 0.05
C_STRENGTH = 0.01
A_STRENGTH = 0.05

DT = 1
STEPS = 5000


def generate_boids(n=N, speed=SPEED, device=DEVICE):
    """
    Generate initial positions and velocities for boids.

    Args:
        n (int): Number of boids to generate.
        speed (float): Target constant speed for all boids.
        device (str): Torch device to place tensors ("cpu" or "cuda").

    Returns:
        tuple:
            pos (Tensor[N, 3]): Random positions within simulation bounds.
            vel (Tensor[N, 3]): Normalized velocities with fixed magnitude = speed.
    """
    bounds = torch.tensor([WIDTH, HEIGHT, DEPTH], dtype=torch.float32, device=device)
    pos = torch.rand((n, 3), device=device) * bounds
    vel = torch.randn((n, 3), device=device)
    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed
    return pos, vel


def cohesion(
    pos, vel, perception_radius=C_RADIUS, cohesion_strength=C_STRENGTH, speed=SPEED
):
    """
    Adjusts boid velocities to steer toward the average position of nearby neighbors (cohesion rule in flocking behavior).

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing boid velocities.
        perception_radius (float, optional): Distance within which neighbors are considered for cohesion.
            Defaults to C_RADIUS.
        cohesion_strength (float, optional): Weight factor controlling how strongly boids steer toward the average neighbor position.
            Defaults to C_STRENGTH.
        speed (float, optional): Target constant speed for all boids after normalization.
            Defaults to SPEED.

    Returns:
        torch.Tensor: Updated velocities of shape (N, 3) after applying cohesion.
    """
    N = pos.shape[0]

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = torch.norm(diff, dim=2)

    mask = (dist < perception_radius) & (dist > 0)

    neighbor_pos_sum = torch.matmul(mask.float(), pos)
    neighbor_count = mask.sum(dim=1, keepdim=True)
    neighbor_count = torch.clamp(neighbor_count, min=1.0)

    avg_neighbor_pos = neighbor_pos_sum / neighbor_count

    vel = vel + (avg_neighbor_pos - pos) * cohesion_strength

    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed

    return vel


def separation(
    pos, vel, min_distance=S_RADIUS, separation_strength=S_STRENGTH, speed=SPEED
):
    """
    Adjusts boid velocities to steer away from nearby neighbors that are too close (separation rule in flocking behavior).

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing boid velocities.
        min_distance (float, optional): Minimum allowed distance between boids.
            Neighbors closer than this threshold trigger repulsion.
            Defaults to S_RADIUS.
        separation_strength (float, optional): Weight factor controlling how strongly boids steer away from close neighbors.
            Defaults to S_STRENGTH.
        speed (float, optional): Target constant speed for all boids after normalization.
            Defaults to SPEED.

    Returns:
        torch.Tensor: Updated velocities of shape (N, 3) after applying separation.
    """
    N = pos.shape[0]

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = torch.norm(diff, dim=2)

    mask = (dist < min_distance) & (dist > 0)

    repulsion = torch.sum(diff * mask.unsqueeze(2), dim=1)

    vel = vel + repulsion * separation_strength

    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed

    return vel


def alignment(
    pos, vel, perception_radius=A_RADIUS, alignment_strength=A_STRENGTH, speed=SPEED
):
    """
    Adjusts boid velocities to align with the average heading of nearby neighbors (alignment rule in flocking behavior).

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing boid velocities.
        perception_radius (float, optional): Distance within which neighbors are considered for alignment.
            Defaults to A_RADIUS.
        alignment_strength (float, optional): Weight factor controlling how strongly boids steer toward the average neighbor velocity.
            Defaults to A_STRENGTH.
        speed (float, optional): Target constant speed for all boids after normalization.
            Defaults to SPEED.

    Returns:
        torch.Tensor: Updated velocities of shape (N, 3) after applying alignment.
    """
    N = pos.shape[0]

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = torch.norm(diff, dim=2)

    mask = (dist < perception_radius) & (dist > 0)

    neighbor_vel_sum = torch.matmul(mask.float(), vel)
    neighbor_count = mask.sum(dim=1, keepdim=True)
    neighbor_count = torch.clamp(neighbor_count, min=1.0)

    avg_neighbor_vel = neighbor_vel_sum / neighbor_count

    vel = vel + (avg_neighbor_vel - vel) * alignment_strength

    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed

    return vel


def step(pos, vel, dt=DT):
    """
    Advances the boid simulation by one time step.

    This function applies the three flocking rules (alignment, cohesion, separation) to update boid velocities, then moves boids forward in time by `dt`. It also enforces boundary conditions by bouncing boids off the walls of the simulation box when they reach or exceed the limits.

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing boid velocities.
        dt (float, optional): Time step for position updates. Defaults to DT.

    Returns:
        tuple:
            pos (torch.Tensor): Updated positions of shape (N, 3).
            vel (torch.Tensor): Updated velocities of shape (N, 3).
    """
    vel = alignment(
        pos, vel, perception_radius=50.0, alignment_strength=0.05, speed=2.0
    )
    vel = cohesion(pos, vel, perception_radius=50.0, cohesion_strength=0.01, speed=2.0)
    vel = separation(pos, vel, min_distance=20.0, separation_strength=0.05, speed=2.0)

    pos = pos + vel * dt

    mask_x_low = pos[:, 0] < 0
    mask_x_high = pos[:, 0] > WIDTH
    vel[mask_x_low | mask_x_high, 0] *= -1
    pos[:, 0] = pos[:, 0].clamp(0, WIDTH)

    mask_y_low = pos[:, 1] < 0
    mask_y_high = pos[:, 1] > HEIGHT
    vel[mask_y_low | mask_y_high, 1] *= -1
    pos[:, 1] = pos[:, 1].clamp(0, HEIGHT)

    mask_z_low = pos[:, 2] < 0
    mask_z_high = pos[:, 2] > DEPTH
    vel[mask_z_low | mask_z_high, 2] *= -1
    pos[:, 2] = pos[:, 2].clamp(0, DEPTH)

    return pos, vel


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


def tensor_to_lines(pos_tensor, vel_tensor, scale=10.0):
    pos = pos_tensor.detach().cpu().numpy()
    vel = vel_tensor.detach().cpu().numpy()

    start_points = pos
    end_points = pos + vel * scale

    points = np.vstack([start_points, end_points])
    lines = [[i, i + len(pos)] for i in range(len(pos))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in lines])
    return line_set


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


# -----------------------------
# Run once
# -----------------------------
pos, vel = generate_boids()
run_simulation(pos, vel)
