import numpy as np
from noise import pnoise3

def generate_noise_grid(x, y, z, rng, radius, scale=0.1):
    size = (2 * radius) + 1
    # Create a 3D grid
    grid = np.zeros((size, size, size))
    start_point = np.array([x, y, z]) - radius

    for i in range(size):
        for j in range(size):
            for k in range(size):
                pos = start_point + [i, j, k]
                grid[i, j, k] = pnoise3(
                    (pos[0] + rng) * scale,
                    (pos[1] + rng) * scale,
                    (pos[2] + rng) * scale
                )
    return grid
