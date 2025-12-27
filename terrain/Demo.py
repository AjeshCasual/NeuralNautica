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

mc_mesh = mesh

bounds = mc_mesh.bounds  # [[minx, miny, minz], [maxx, maxy, maxz]]

padding = 5.0  # adjust as needed
min_corner = bounds[0] - padding
max_corner = bounds[1] + padding

box_size = max_corner - min_corner
box_center = (max_corner + min_corner) / 2.0
outer_box = trimesh.creation.box(
    extents=box_size, transform=trimesh.transformations.translation_matrix(box_center)
)

combined = mc_mesh + outer_box

combined.export("out_with_box.obj")
