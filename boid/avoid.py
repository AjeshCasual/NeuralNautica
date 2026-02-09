import torch

def get_updated_boid_velocities(positions, velocities, noise_grids, n_rays=100, angle_deg=45, level=0.0, max_dist=5.0):
    """
    Inputs:
        positions: (N, 3) current positions
        velocities: (N, 3) current velocities
        noise_grids: (N, D, D, D) the grids from function above
    Returns:
        updated_vel: (N, 3) the new velocity steered away from obstacles
    """
    N = positions.shape[0]
    device = positions.device

    # 1. Generate & Rotate Fibonacci Rays (N, M, 3)
    # This ensures sensors point where the player is moving
    ray_dirs = _get_batched_fibonacci_rays(velocities, n_rays, angle_deg, device)

    # 2. Define Weights (Center rays are 1.0, peripheral are 0.2)
    weights = torch.linspace(1.0, 0.2, n_rays, device=device).view(1, -1, 1)

    # 3. Sample the "Mathematical Mesh" along rays
    num_steps = 10
    distances = torch.linspace(0.5, max_dist, num_steps, device=device)
    total_avoidance_force = torch.zeros((N, 3), device=device)

    for d in distances:
        # Calculate sample points in local grid UV space [-1, 1]
        # (N, M, 3)
        sample_dirs = ray_dirs * d
        grid_uv = sample_dirs / max_dist # Normalize to detection radius

        # Grid Sample: Samples the (N, D, D, D) noise at the ray points
        sampled_noise = torch.nn.functional.grid_sample(
            noise_grids.unsqueeze(1),
            grid_uv.unsqueeze(2).unsqueeze(2),
            mode='bilinear', align_corners=True
        ).reshape(N, n_rays)

        # Collision Check: Is the noise value above the "Marching Cubes" level?
        hits = (sampled_noise > level).float().unsqueeze(-1) # (N, M, 1)

        # Add repulsion: Push away from the ray direction
        # Urgency is 1/d (closer obstacles = harder turn)
        total_avoidance_force += (ray_dirs * -1.0) * hits * weights * (1.0 / d)

    # 4. Integrate with current velocity
    # We add the avoidance force to the original velocity
    # You can multiply avoidance_force by a 'sensitivity' constant here
    updated_vel = velocities + total_avoidance_force

    return updated_vel
