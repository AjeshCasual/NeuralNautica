from generate import generate_boids
from visualize import run_simulation

pos, vel = generate_boids()
run_simulation(pos, vel)
