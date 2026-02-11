def get_terrain_geometry(pos, gen, radius, threshold=0.15):
    """
    pos: (N, 3) tensor
    gen: (N, D, D, D) tensor
    """
    N, D, _, _ = gen.shape
    device = gen.device

    # Create a coordinate grid once to map voxel indices to world space
    # This matches the start_offsets logic in your generator
    coords = torch.arange(D, device=device) - radius
    gz, gy, gx = torch.meshgrid(coords, coords, coords, indexing='ij')
    local_grid = torch.stack([gx, gy, gz], dim=-1) # (D, D, D, 3)

    # Find all voxels across all bots that are "solid"
    mask = gen > threshold

    # Get world positions of solid voxels
    # We add the bot's position to the local grid offsets
    all_terrain_points = []

    for i in range(N):
        solid_indices = mask[i]
        if torch.any(solid_indices):
            world_voxels = local_grid[solid_indices] + pos[i]
            all_terrain_points.append(world_voxels)

    if not all_terrain_points:
        return o3d.geometry.PointCloud()

    # Combine and convert to Open3D
    combined_pts = torch.cat(all_terrain_points, dim=0).detach().cpu().numpy()

    terrain_pcd = o3d.geometry.PointCloud()
    terrain_pcd.points = o3d.utility.Vector3dVector(combined_pts)
    # Color terrain gray/brown
    terrain_pcd.paint_uniform_color([0.4, 0.3, 0.2])

    # Optional: Downsample to keep it fast
    terrain_pcd = terrain_pcd.voxel_down_sample(voxel_size=0.1)

    return terrain_pcd
