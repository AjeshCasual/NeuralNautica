import open3d as o3d
import numpy as np

def get_boids_and_rays_geometry(pos, vel, rays, ray_intense, radius, threshold=0.15):
    """
    pos: (N, 3) tensor
    rays: (N, n_rays, 3) tensor (directions)
    ray_intense: (N, n_rays) tensor
    """
    # Convert to numpy for Open3D
    pos_np = pos.detach().cpu().numpy()
    rays_np = rays.detach().cpu().numpy()
    intense_np = ray_intense.detach().cpu().numpy()

    N, n_rays, _ = rays_np.shape

    # --- 1. Boids Point Cloud ---
    boids_pcd = o3d.geometry.PointCloud()
    boids_pcd.points = o3d.utility.Vector3dVector(pos_np)
    # Color boids blue
    boids_pcd.paint_uniform_color([0.2, 0.5, 1.0])

    # --- 2. Rays Line Set ---
    # Every line needs two points: [start, end]
    # We create N * n_rays * 2 points
    ray_endpoints = pos_np[:, np.newaxis, :] + (rays_np * radius * 0.8)

    # Flatten everything for Open3D
    all_points = []
    lines = []
    colors = []

    for i in range(N):
        for r in range(n_rays):
            start_idx = len(all_points)
            all_points.append(pos_np[i])
            all_points.append(ray_endpoints[i, r])

            lines.append([start_idx, start_idx + 1])

            # Color logic: Red if intense (hit), Green if clear
            if intense_np[i, r] > 0:
                colors.append([1.0, 0.0, 0.0]) # Red
            else:
                colors.append([0.0, 1.0, 0.0]) # Green

    ray_lines = o3d.geometry.LineSet()
    ray_lines.points = o3d.utility.Vector3dVector(np.array(all_points))
    ray_lines.lines = o3d.utility.Vector2iVector(np.array(lines))
    ray_lines.colors = o3d.utility.Vector3dVector(np.array(colors))

    return boids_pcd, ray_lines
