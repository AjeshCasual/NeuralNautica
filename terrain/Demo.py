import numpy as np
import trimesh
from noise import pnoise3
from skimage import measure

SIZE = 32
SCALE = 50
THRESHOLD = 0.1
OUTPUT_FILE = "out.obj"

grid = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)

for x in range(SIZE):
    for y in range(SIZE):
        for z in range(SIZE):
            grid[x, y, z] = pnoise3(x / SCALE, y / SCALE, z / SCALE)


verts, faces, normals, values = measure.marching_cubes(grid, level=THRESHOLD)

mesh = trimesh.Trimesh(vertices=verts, faces=faces)

mesh.export(OUTPUT_FILE)
