import numpy as np
import trimesh
from noise import pnoise3
from skimage import measure

size = 32
grid = np.zeros((size, size, size), dtype=np.float32)

for x in range(size):
    for y in range(size):
        for z in range(size):
            grid[x, y, z] = pnoise3(x / 50, y / 50, z / 50)


verts, faces, normals, values = measure.marching_cubes(grid, level=0.3)

mesh = trimesh.Trimesh(vertices=verts, faces=faces)

mesh.export("out.obj")

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
