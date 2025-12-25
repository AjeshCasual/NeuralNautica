import numpy as np
import open3d as o3d
import torch

# Simulation bounds
WIDTH, HEIGHT, DEPTH = 1000, 1000, 1000
N = 200  # number of boids (keep small for visualization)
device = "cuda"


# -----------------------------
# Generate boids on GPU
# -----------------------------
def generate_boids(n=N, speed=2.0, device=device):
    bounds = torch.tensor([WIDTH, HEIGHT, DEPTH], dtype=torch.float32, device=device)
    pos = torch.rand((n, 3), device=device) * bounds
    vel = torch.randn((n, 3), device=device)
    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed
    return pos, vel


# -----------------------------
# Bounding box
# -----------------------------
def create_bounding_box(xmin=0, ymin=0, zmin=0, xmax=WIDTH, ymax=HEIGHT, zmax=DEPTH):
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


# -----------------------------
# Convert tensors to arrows
# -----------------------------
def tensor_to_arrows(pos_tensor, vel_tensor, length=10.0, radius=1.0):
    arrows = []
    pos = pos_tensor.detach().cpu().numpy()
    vel = vel_tensor.detach().cpu().numpy()

    for p, v in zip(pos, vel):
        direction = v / (np.linalg.norm(v) + 1e-8)

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=radius,
            cone_radius=radius * 1.5,
            cylinder_height=length * 0.7,
            cone_height=length * 0.3,
        )
        arrow.paint_uniform_color([0.2, 0.8, 0.2])

        # Rotate arrow from +Z to velocity direction
        z_axis = np.array([0, 0, 1])
        v_cross = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)
        s = np.linalg.norm(v_cross)
        if s > 1e-8:
            vx = np.array(
                [
                    [0, -v_cross[2], v_cross[1]],
                    [v_cross[2], 0, -v_cross[0]],
                    [-v_cross[1], v_cross[0], 0],
                ]
            )
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
            arrow.rotate(R, center=(0, 0, 0))

        # Translate to boid position
        arrow.translate(p)
        arrows.append(arrow)

    return arrows


# -----------------------------
# Visualization
# -----------------------------
def view_boids_with_arrows(pos_tensor, vel_tensor):
    bbox = create_bounding_box(0, 0, 0, WIDTH, HEIGHT, DEPTH)
    arrows = tensor_to_arrows(pos_tensor, vel_tensor)
    o3d.visualization.draw_geometries([bbox] + arrows)


# -----------------------------
# Run once
# -----------------------------
pos, vel = generate_boids()
view_boids_with_arrows(pos, vel)
