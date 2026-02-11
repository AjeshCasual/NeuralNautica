import torch
from rules import *
from var import WIDTH, HEIGHT, DEPTH, DT
import avoid
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from terrain.perlin_cuda import generate_batched_noise_grids

def step(pos, vel, dt=DT):
    # 1. Social Steering (The Flocking Rules)
    vel = alignment(pos, vel)
    vel = cohesion(pos, vel)
    vel = separation(pos, vel)

    # 2. Perception Setup
    # radius=5 or 6 gives the bots a "vision bubble"
    # scale=0.1 determines how "stretched" the terrain is
    view_dist = 6

    # Passing 42 directly works because your function expands it!
    # Using the same number for all ensures they share the same world.
    gen = generate_batched_noise_grids(pos, rng_seeds=42, radius=view_dist, scale=0.1)

    # 3. Terrain Avoidance
    # n_rays: more rays = smoother detection but higher GPU cost
    # threshold: what density they consider "solid" (0.1 - 0.3 is typical)
    vel, rays, ray_intense = avoid.avoid_terrain_with_viz(
        pos, vel, gen, radius=view_dist, n_rays=15, threshold=0.15
    )

    # 4. Movement
    pos = pos + vel * dt

    return pos, vel, gen, rays, ray_intense
