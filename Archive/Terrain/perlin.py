import torch

def generate_noise_grid(x, y, z, rng, radius, scale=0.1, device='cuda'):
    """
    GPU version of generate_noise_grid.
    Returns a (Size, Size, Size) tensor on the GPU.
    """
    size = int((2 * radius) + 1)

    # Create the coordinate grid (Size, Size, Size, 3)
    coords = torch.stack(torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing='ij'
    ), dim=-1).float()

    # Offset by world position and rng
    start_point = torch.tensor([x, y, z], device=device) - radius
    world_coords = (coords + start_point + rng) * scale

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
