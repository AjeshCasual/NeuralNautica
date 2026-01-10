import torch
from rules import *
from var import WIDTH, HEIGHT, DEPTH, DT

def step(pos, vel, dt=DT):
    """
    Advances the boid simulation by one time step.
    This function applies the three flocking rules (alignment, cohesion, separation) to update boid velocities, then moves boids forward in time by `dt`. It also enforces boundary conditions by bouncing boids off the walls of the simulation box when they reach or exceed the limits.
    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing boid velocities.
        dt (float, optional): Time step for position updates. Defaults to DT.

    Returns:
        tuple:
            pos (torch.Tensor): Updated positions of shape (N, 3).
            vel (torch.Tensor): Updated velocities of shape (N, 3).
    """
    vel = alignment(pos, vel)
    vel = cohesion(pos, vel)
    vel = separation(pos, vel)

    pos = pos + vel * dt

    mask_x_low = pos[:, 0] < 0
    mask_x_high = pos[:, 0] > WIDTH
    vel[mask_x_low | mask_x_high, 0] *= -1
    pos[:, 0] = pos[:, 0].clamp(0, WIDTH)

    mask_y_low = pos[:, 1] < 0
    mask_y_high = pos[:, 1] > HEIGHT
    vel[mask_y_low | mask_y_high, 1] *= -1
    pos[:, 1] = pos[:, 1].clamp(0, HEIGHT)

    mask_z_low = pos[:, 2] < 0
    mask_z_high = pos[:, 2] > DEPTH
    vel[mask_z_low | mask_z_high, 2] *= -1
    pos[:, 2] = pos[:, 2].clamp(0, DEPTH)

    return pos, vel
