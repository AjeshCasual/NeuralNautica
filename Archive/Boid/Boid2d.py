import math
import random

import numpy as np
import open3d as o3d
from Vector2D import Vector2D

WIDTH = 1000
HEIGHT = 800


class Boid2D:
    def __init__(self, pos: Vector2D, vel: Vector2D, speed: float, limit: float):
        self.pos = pos
        self.vel = vel
        self.speed = speed
        self.limit = limit

    def separation(self, boids, radius: float, strength: float):
        steer = Vector2D(0.0, 0.0)
        count = 0

        for other in boids:
            dx = self.pos.x - other.pos.x
            dy = self.pos.y - other.pos.y
            dist = math.sqrt(dx * dx + dy * dy)

            if 0 < dist < radius:
                steer.x += dx / dist
                steer.y += dy / dist
                count += 1

        if count > 0:
            steer.x /= count
            steer.y /= count

            # apply steering to velocity
            self.vel.x += steer.x * strength
            self.vel.y += steer.y * strength

            self.vel = self.vel.normalize() * self.speed

    def alignment(self, boids, radius: float, strength: float):
        vel_avg = Vector2D(0.0, 0.0)
        count = 0
        for other in boids:
            dx = self.pos.x - other.pos.x
            dy = self.pos.y - other.pos.y
            dist = math.sqrt(dx * dx + dy * dy)

            if 0 < dist < radius:
                vel_avg += other.vel
                count += 1

        if count > 0:
            vel_avg = vel_avg * (1.0 / count)
            self.vel.x += (vel_avg.x - self.vel.x) * strength
            self.vel.y += (vel_avg.y - self.vel.y) * strength
            self.vel = self.vel.normalize() * self.speed

    def cohesion(self, boids, radius: float, strength: float):
        pos_avg = Vector2D(0.0, 0.0)
        count = 0
        for other in boids:
            dx = self.pos.x - other.pos.x
            dy = self.pos.y - other.pos.y
            dist = math.sqrt(dx * dx + dy * dy)

            if 0 < dist < radius:
                pos_avg += other.pos
                count += 1

        if count > 0:
            pos_avg = pos_avg * (1.0 / count)
            self.vel.x += (pos_avg.x - self.pos.x) * strength
            self.vel.y += (pos_avg.y - self.pos.y) * strength
            self.vel = self.vel.normalize() * self.speed

    def step(self, dt):
        self.pos += self.vel * dt

        if self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.x > WIDTH:
            self.pos.x = 0
        if self.pos.y < 0:
            self.pos.y = HEIGHT
        if self.pos.y > HEIGHT:
            self.pos.y = 0


def create_bounding_box(xmin=0, ymin=0, xmax=1000, ymax=600):
    # Define 4 corners in XY plane (z=0)
    points = np.array(
        [[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0]]
    )

    # Define edges between corners
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optional: color the lines (red)
    colors = [[1, 0, 0] for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def boids_to_pointcloud(boids):
    points = np.array([[b.pos.x, b.pos.y, 0] for b in boids], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def run_simulation(boids, steps=200000, dt=1):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add boids as point cloud
    pcd = boids_to_pointcloud(boids)
    vis.add_geometry(pcd)

    # Add bounding box
    bbox = create_bounding_box(0, 0, WIDTH, HEIGHT)
    vis.add_geometry(bbox)

    for _ in range(steps):
        for b in boids:
            b.separation(boids, radius=8, strength=0.05)
            b.alignment(boids, radius=40, strength=0.05)
            b.cohesion(boids, radius=40, strength=0.0005)
            b.step(dt)

        points = np.array([[b.pos.x, b.pos.y, 0] for b in boids], dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(points)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


boids = [
    Boid2D(
        pos=Vector2D(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
        vel=Vector2D(random.uniform(-1, 1), random.uniform(-1, 1)),
        speed=2,
        limit=2,
    )
    for _ in range(100)
]

run_simulation(boids, steps=20000)
