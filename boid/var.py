"""
Boid Simulation Configuration

Global constants used throughout the boid simulation, including simulation space dimensions, number of boids, device settings, and flocking rule parameters.

Constants:
    WIDTH (int): Width of the 3D simulation space.
    HEIGHT (int): Height of the 3D simulation space.
    DEPTH (int): Depth of the 3D simulation space.

    N (int): Number of boids in the simulation.

    DEVICE (str): Torch device to use ("cpu" or "cuda").

    SPEED (float): Target constant speed for all boids.

    S_RADIUS (float): Perception radius for separation (neighbors considered).
    C_RADIUS (float): Perception radius for cohesion (neighbors considered).
    A_RADIUS (float): Perception radius for alignment (neighbors considered).

    S_STRENGTH (float): Strength of separation steering.
    C_STRENGTH (float): Strength of cohesion steering.
    A_STRENGTH (float): Strength of alignment steering.

    DT (float): Time step for position updates.
    STEPS (int): Number of simulation steps to run.
"""


WIDTH = 1000
HEIGHT = 1000
DEPTH = 1000
N = 5000
DEVICE = "cuda"
SPEED = 2.0
S_RADIUS = 20.0
C_RADIUS = 50.0
A_RADIUS = 50.0
S_STRENGTH = 0.05
C_STRENGTH = 0.01
A_STRENGTH = 0.05
DT = 1
STEPS = 5000000
