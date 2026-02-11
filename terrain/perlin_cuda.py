import torch

def generate_batched_noise_grids(positions, rng_seeds, radius, scale=0.1, device='cuda'):
    """
    Generates a unique noise volume for N players at once.
    Returns: Tensor of shape (N, D, D, D) where D = 2*radius + 1
    """
    N = positions.shape[0]
    D = int((2 * radius) + 1)

    # 1. Create a local coordinate grid [D, D, D, 3]
    # This represents the relative offsets from the player's center
    coords = torch.stack(torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(D, device=device),
        torch.arange(D, device=device),
        indexing='ij'
    ), dim=-1).float()

    # 2. Shift and Scale to World Noise Space
    # We expand coords to (N, D, D, D, 3) and add player positions
    start_offsets = (positions - radius).view(N, 1, 1, 1, 3)
    # Add the unique RNG seed per player to ensure they aren't all in the same noise# Ensure rng_seeds is a tensor on the correct device
    if not isinstance(rng_seeds, torch.Tensor):
        rng_seeds = torch.tensor(rng_seeds, device=device).expand(N)

    # Now .view(N, 1, 1, 1, 1) will work
    world_coords = (coords.unsqueeze(0) + start_offsets + rng_seeds.view(N, 1, 1, 1, 1)) * scale

    # 3. Compute Perlin Noise (using the vectorized function from earlier)
    # Returns (N, D, D, D)
    return perlin_3d_pytorch(world_coords, device)

def perlin_3d_pytorch(coords, device):
    """Vectorized 3D Perlin Noise implementation for PyTorch."""
    # Split coordinates into integer and fractional parts
    p0 = torch.floor(coords).long()
    p1 = p0 + 1
    f = coords - p0.float()

    # Fade function: 6t^5 - 15t^4 + 10t^3
    fade_f = f * f * f * (f * (f * 6 - 15) + 10)

    # Hash function to get gradients (simplified for speed but consistent)
    def hash_coords(p):
        # A simple large-prime hash for GPU
        h = p[..., 0] * 127 + p[..., 1] * 31337 + p[..., 2] * 47
        return (torch.sin(h.float()) * 43758.5453).frac()

    # Get gradients at 8 corners of the cube
    n000 = hash_coords(p0) * 2 - 1
    n100 = hash_coords(torch.stack([p1[...,0], p0[...,1], p0[...,2]], -1)) * 2 - 1
    n010 = hash_coords(torch.stack([p0[...,0], p1[...,1], p0[...,2]], -1)) * 2 - 1
    n110 = hash_coords(torch.stack([p1[...,0], p1[...,1], p0[...,2]], -1)) * 2 - 1
    n001 = hash_coords(torch.stack([p0[...,0], p0[...,1], p1[...,2]], -1)) * 2 - 1
    n101 = hash_coords(torch.stack([p1[...,0], p0[...,1], p1[...,2]], -1)) * 2 - 1
    n011 = hash_coords(torch.stack([p0[...,0], p1[...,1], p1[...,2]], -1)) * 2 - 1
    n111 = hash_coords(p1) * 2 - 1

    # Trilinear interpolation
    nx00 = torch.lerp(n000, n100, fade_f[..., 0])
    nx01 = torch.lerp(n001, n101, fade_f[..., 0])
    nx10 = torch.lerp(n010, n110, fade_f[..., 0])
    nx11 = torch.lerp(n011, n111, fade_f[..., 0])

    nxy0 = torch.lerp(nx00, nx10, fade_f[..., 1])
    nxy1 = torch.lerp(nx01, nx11, fade_f[..., 1])

    return torch.lerp(nxy0, nxy1, fade_f[..., 2])
