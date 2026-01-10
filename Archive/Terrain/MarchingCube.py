import sys

import numpy as np
import pygame
import trimesh
from noise import pnoise3
from skimage import measure

# 1. Generate Perlin noise grid
size = 64  # keep smaller for speed in pygame
grid = np.zeros((size, size, size), dtype=np.float32)

for x in range(size):
    for y in range(size):
        for z in range(size):
            grid[x, y, z] = pnoise3(
                x / 40, y / 40, z / 40, octaves=3, persistence=0.5, lacunarity=2.0
            )

# 2. Extract mesh
verts, faces, normals, values = measure.marching_cubes(grid, level=0.2)


# 3. Simple projection function (orthographic)
def project(v, scale=4, offset=(400, 300)):
    """Project 3D vertex v into 2D pygame coordinates."""
    x, y, z = v
    return int(x * scale + offset[0]), int(y * scale + offset[1])


# 4. Setup pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# 5. Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    # Draw mesh faces
    for f in faces:
        pts = [project(verts[idx]) for idx in f]
        pygame.draw.polygon(screen, (0, 200, 100), pts, 1)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
