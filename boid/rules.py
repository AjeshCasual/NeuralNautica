import torch
from var import SPEED, S_RADIUS, C_RADIUS, A_RADIUS, S_STRENGTH, C_STRENGTH, A_STRENGTH

def cohesion(
    pos, vel, perception_radius=C_RADIUS, cohesion_strength=C_STRENGTH, speed=SPEED
):
    """
    Adjusts boid velocities to steer toward the average position of nearby neighbors (cohesion rule in flocking behavior).

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing boid velocities.
        perception_radius (float, optional): Distance within which neighbors are considered for cohesion.
            Defaults to C_RADIUS.
        cohesion_strength (float, optional): Weight factor controlling how strongly boids steer toward the average neighbor position.
            Defaults to C_STRENGTH.
        speed (float, optional): Target constant speed for all boids after normalization.
            Defaults to SPEED.

    Returns:
        torch.Tensor: Updated velocities of shape (N, 3) after applying cohesion.
    """
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = torch.norm(diff, dim=2)
    mask = (dist < perception_radius) & (dist > 0)
    neighbor_pos_sum = torch.matmul(mask.float(), pos)
    neighbor_count = mask.sum(dim=1, keepdim=True)
    neighbor_count = torch.clamp(neighbor_count, min=1.0)
    avg_neighbor_pos = neighbor_pos_sum / neighbor_count
    vel = vel + (avg_neighbor_pos - pos) * cohesion_strength
    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed

    return vel


def separation(
    pos, vel, min_distance=S_RADIUS, separation_strength=S_STRENGTH, speed=SPEED
):
    """
    Adjusts boid velocities to steer away from nearby neighbors that are too close (separation rule in flocking behavior).

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing boid velocities.
        min_distance (float, optional): Minimum allowed distance between boids.
            Neighbors closer than this threshold trigger repulsion.
            Defaults to S_RADIUS.
        separation_strength (float, optional): Weight factor controlling how strongly boids steer away from close neighbors.
            Defaults to S_STRENGTH.
        speed (float, optional): Target constant speed for all boids after normalization.
            Defaults to SPEED.

    Returns:
        torch.Tensor: Updated velocities of shape (N, 3) after applying separation.
    """
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = torch.norm(diff, dim=2)
    mask = (dist < min_distance) & (dist > 0)
    repulsion = torch.sum(diff * mask.unsqueeze(2), dim=1)
    vel = vel + repulsion * separation_strength
    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed

    return vel


def alignment(
    pos, vel, perception_radius=A_RADIUS, alignment_strength=A_STRENGTH, speed=SPEED
):
    """
    Adjusts boid velocities to align with the average heading of nearby neighbors (alignment rule in flocking behavior).

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) representing boid positions.
        vel (torch.Tensor): Tensor of shape (N, 3) representing boid velocities.
        perception_radius (float, optional): Distance within which neighbors are considered
            for alignment. Defaults to A_RADIUS.
        alignment_strength (float, optional): Weight factor controlling how strongly boids
            steer toward the average neighbor velocity. Defaults to A_STRENGTH.
        speed (float, optional): Target constant speed for all boids after normalization.
            Defaults to SPEED.

    Returns:
        torch.Tensor: Updated velocities of shape (N, 3) after applying alignment.
    """
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = torch.norm(diff, dim=2)
    mask = (dist < perception_radius) & (dist > 0)
    neighbor_vel_sum = torch.matmul(mask.float(), vel)
    neighbor_count = mask.sum(dim=1, keepdim=True)
    neighbor_count = torch.clamp(neighbor_count, min=1.0)
    avg_neighbor_vel = neighbor_vel_sum / neighbor_count
    vel = vel + (avg_neighbor_vel - vel) * alignment_strength
    vel = vel / (torch.norm(vel, dim=1, keepdim=True) + 1e-8) * speed

    return vel
