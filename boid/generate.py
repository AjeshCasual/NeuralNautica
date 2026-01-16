import torch
from var import N,SPEED,DEVICE,WIDTH, HEIGHT, DEPTH

def generate_boids(n=N, speed=SPEED, device=DEVICE,width=WIDTH,height=HEIGHT,depth=DEPTH):
    """
    Generate initial positions and velocities for boids.

    Args:
        n (int): Number of boids to generate.
        speed (float): Target constant speed for all boids.
        device (str): Torch device to place tensors ("cpu" or "cuda").

    Returns:
        tuple:
            pos (Tensor[N, 3]): Random positions within simulation bounds.
            vel (Tensor[N, 3]): Normalized velocities with fixed magnitude = speed.
    """
    bounds = torch.tensor([width,height,depth], dtype=torch.float32, device=device)
    pos = torch.rand((n, 3), device=device) * bounds
    vel = torch.randn((n, 3), device=device)
    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed
    return pos, vel
