import torch
import torch.nn.functional as F

def avoid_terrain_with_viz(positions, velocities, noise_grids, radius, n_rays=100, angle_deg=45, threshold=0.1):
    N = positions.shape[0]
    device = positions.device
    D = noise_grids.shape[1]
    center_idx = radius

    # 1. Generate Fibonacci Cone (Local Space)
    indices = torch.arange(n_rays, device=device)
    phi = (1 + 5**0.5) / 2
    cos_theta_max = torch.cos(torch.tensor(angle_deg * (3.14159 / 180.0)))

    z = 1.0 - (indices / float(n_rays - 1)) * (1.0 - cos_theta_max)
    radius_at_z = torch.sqrt(1.0 - z*z)
    theta = 2 * 3.14159 * indices / phi
    local_rays = torch.stack([radius_at_z * torch.cos(theta), radius_at_z * torch.sin(theta), z], dim=-1)

    # 2. Rotate Rays to align with Velocity
    # We find the rotation from +Z (0,0,1) to the velocity vector
    # Ensure forward is (N, 3)
    forward = F.normalize(velocities, dim=-1)

    # 1. Create a stable 'up' vector (N, 3)
    # Force the dtype to be float32 at the start
    up = torch.tensor([0.0, 1.0, 0.0], device=device).expand(N, 3).clone()

    # This will now work because both sides are floats
    is_vertical = torch.abs(forward[:, 1]) > 0.99
    up[is_vertical] = torch.tensor([1.0, 0.0, 0.0], device=device)

    # 3. Calculate Right and True Up
    # Use .contiguous() to ensure memory is packed correctly for the cross product
    right = F.normalize(torch.cross(up, forward, dim=-1), dim=-1)
    true_up = torch.cross(forward, right, dim=-1)

    # 4. Rotation Matrix [Right | True_Up | Forward]
    # Shape: (N, 3, 3)
    rot_matrix = torch.stack([right, true_up, forward], dim=-1)

    # Transform local rays to world rays: (N, n_rays, 3)
    world_rays = torch.bmm(local_rays.expand(N, n_rays, 3), rot_matrix.transpose(-2, -1))

    # 3. Ray Marching & Weighting
    steps = torch.linspace(0.2, 1.0, steps=5, device=device) * radius
    repulsion_vec = torch.zeros((N, 3), device=device)

    # Track which rays hit terrain for visualization (optional)
    ray_intensities = torch.zeros((N, n_rays), device=device)

    for step in steps:
        grid_pos = (world_rays * step) + center_idx
        grid_pos = grid_pos.long().clamp(0, D - 1)

        n_idx = torch.arange(N, device=device).view(N, 1)
        sampled_noise = noise_grids[n_idx, grid_pos[..., 0], grid_pos[..., 1], grid_pos[..., 2]]

        intensity = F.relu(sampled_noise - threshold)
        weight = intensity / (step ** 2)

        repulsion_vec -= (weight.unsqueeze(-1) * world_rays).sum(dim=1)
        ray_intensities += intensity # For viz: which rays are "red"

    # 4. Final Velocity
    avoidance_strength = 0.8
    final_vel = velocities + (repulsion_vec * avoidance_strength)
    speed = velocities.norm(dim=-1, keepdim=True)

    return F.normalize(final_vel, dim=-1) * speed, world_rays, ray_intensities
