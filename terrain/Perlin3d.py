import numpy as np
import open3d as o3d
import torch


# --- Perlin Noise 3D ---
def perlin_3d(shape=(64, 64, 64), res=(8, 8, 8)):
    """Generate 3D Perlin noise using PyTorch tensors."""

    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, res[0] + 1, dtype=torch.float32),
            torch.arange(0, res[1] + 1, dtype=torch.float32),
            torch.arange(0, res[2] + 1, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    )

    # Random gradients
    gradients = torch.randn(res[0] + 1, res[1] + 1, res[2] + 1, 3)
    gradients = gradients / torch.norm(gradients, dim=-1, keepdim=True)

    # Coordinates
    lin = [torch.linspace(0, res[i], shape[i], dtype=torch.float32) for i in range(3)]
    coords = torch.stack(torch.meshgrid(*lin, indexing="ij"), dim=-1)
    g000 = gradients[
        coords[..., 0].long(), coords[..., 1].long(), coords[..., 2].long()
    ]

    # Fractional part
    frac = coords - coords.floor()
    u, v, w = f(frac[..., 0]), f(frac[..., 1]), f(frac[..., 2])

    # Dot products with gradient vectors
    def dot_grid(ix, iy, iz, fx, fy, fz):
        g = gradients[ix, iy, iz]
        return fx * g[..., 0] + fy * g[..., 1] + fz * g[..., 2]

    x0 = coords[..., 0].long()
    y0 = coords[..., 1].long()
    z0 = coords[..., 2].long()

    x1 = torch.clamp(x0 + 1, max=res[0])
    y1 = torch.clamp(y0 + 1, max=res[1])
    z1 = torch.clamp(z0 + 1, max=res[2])

    n000 = dot_grid(x0, y0, z0, frac[..., 0], frac[..., 1], frac[..., 2])
    n100 = dot_grid(x1, y0, z0, frac[..., 0] - 1, frac[..., 1], frac[..., 2])
    n010 = dot_grid(x0, y1, z0, frac[..., 0], frac[..., 1] - 1, frac[..., 2])
    n110 = dot_grid(x1, y1, z0, frac[..., 0] - 1, frac[..., 1] - 1, frac[..., 2])
    n001 = dot_grid(x0, y0, z1, frac[..., 0], frac[..., 1], frac[..., 2] - 1)
    n101 = dot_grid(x1, y0, z1, frac[..., 0] - 1, frac[..., 1], frac[..., 2] - 1)
    n011 = dot_grid(x0, y1, z1, frac[..., 0], frac[..., 1] - 1, frac[..., 2] - 1)
    n111 = dot_grid(x1, y1, z1, frac[..., 0] - 1, frac[..., 1] - 1, frac[..., 2] - 1)

    # Interpolation
    nx00 = n000 * (1 - u) + n100 * u
    nx10 = n010 * (1 - u) + n110 * u
    nx01 = n001 * (1 - u) + n101 * u
    nx11 = n011 * (1 - u) + n111 * u

    nxy0 = nx00 * (1 - v) + nx10 * v
    nxy1 = nx01 * (1 - v) + nx11 * v

    nxyz = nxy0 * (1 - w) + nxy1 * w
    return nxyz


# --- Generate Perlin Noise Volume ---
shape = (64, 64, 64)
noise = perlin_3d(shape, res=(8, 8, 8))

# Normalize to [0,1]
volume = (noise - noise.min()) / (noise.max() - noise.min())
volume_np = volume.numpy()

# --- Visualize with Open3D ---
# Convert to voxel grid (threshold at 0.5)
voxels = []
for x in range(shape[0]):
    for y in range(shape[1]):
        for z in range(shape[2]):
            if volume_np[x, y, z] > 0.5:
                voxels.append([x, y, z])

# Create voxel grid
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(voxels))

# Visualize
o3d.visualization.draw_geometries([pcd])
