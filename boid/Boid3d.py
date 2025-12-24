import math
import random

import numpy as np
import open3d as o3d
from Vector3D import Vector3D

WIDTH = 1000
HEIGHT = 1000
DEPTH = 1000


class Boid3D:
    def __init__(self, pos: Vector3D, vel: Vector3D, speed: float, limit: float):
        self.pos = pos
        self.vel = vel
        self.speed = speed
        self.limit = limit

    def separation(self, boids, radius: float, strength: float):
        steer = Vector3D(0.0, 0.0, 0.0)
        count = 0

        for other in boids:
            dx = self.pos.x - other.pos.x
            dy = self.pos.y - other.pos.y
            dz = self.pos.z - other.pos.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if 0 < dist < radius:
                steer.x += dx / dist
                steer.y += dy / dist
                steer.z += dz / dist
                count += 1

        if count > 0:
            steer.x /= count
            steer.y /= count
            steer.z /= count

            # apply steering to velocity
            self.vel.x += steer.x * strength
            self.vel.y += steer.y * strength
            self.vel.z += steer.z * strength

            self.vel = self.vel.normalize() * self.speed

    def alignment(self, boids, radius: float, strength: float):
        vel_avg = Vector3D(0.0, 0.0, 0.0)
        count = 0
        for other in boids:
            dx = self.pos.x - other.pos.x
            dy = self.pos.y - other.pos.y
            dz = self.pos.z - other.pos.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if 0 < dist < radius:
                vel_avg += other.vel
                count += 1

        if count > 0:
            vel_avg = vel_avg * (1.0 / count)
            self.vel.x += (vel_avg.x - self.vel.x) * strength
            self.vel.y += (vel_avg.y - self.vel.y) * strength
            self.vel.z += (vel_avg.z - self.vel.z) * strength
            self.vel = self.vel.normalize() * self.speed

    def cohesion(self, boids, radius: float, strength: float):
        pos_avg = Vector3D(0.0, 0.0, 0.0)
        count = 0
        for other in boids:
            dx = self.pos.x - other.pos.x
            dy = self.pos.y - other.pos.y
            dz = self.pos.z - other.pos.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if 0 < dist < radius:
                pos_avg += other.pos
                count += 1

        if count > 0:
            pos_avg = pos_avg * (1.0 / count)
            self.vel.x += (pos_avg.x - self.pos.x) * strength
            self.vel.y += (pos_avg.y - self.pos.y) * strength
            self.vel.z += (pos_avg.z - self.pos.z) * strength
            self.vel = self.vel.normalize() * self.speed

    def step(self, dt):
        self.pos += self.vel * dt

        self.pos.x %= WIDTH
        self.pos.y %= HEIGHT
        self.pos.z %= DEPTH


def create_bounding_box(xmin=0, ymin=0, zmin=0, xmax=WIDTH, ymax=HEIGHT, zmax=DEPTH):
    # Define 8 corners of the cube
    points = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )

    # Define edges between corners
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom face
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top face
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical edges
    ]

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optional: color the lines (red)
    colors = [[1, 0, 0] for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def boid_to_cone(boid, radius=2.0, height=6.0, color=[0.2, 0.8, 0.2]):
    # Create a cone mesh (default points along +Z)
    cone = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height)
    cone.paint_uniform_color(color)

    # Normalize velocity to get direction
    direction = np.array([boid.vel.x, boid.vel.y, boid.vel.z])
    direction /= np.linalg.norm(direction)

    # Rotate cone from +Z axis to velocity direction
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)
    s = np.linalg.norm(v)
    if s != 0:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        cone.rotate(R, center=(0, 0, 0))

    # Move cone to boid position
    cone.translate([boid.pos.x, boid.pos.y, boid.pos.z])

    return cone


def boids_to_pointcloud(boids):
    points = np.array([[b.pos.x, b.pos.y, b.pos.z] for b in boids], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def run_simulation(boids, steps=200000, dt=1):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add bounding box
    bbox = create_bounding_box(0, 0, 0, WIDTH, HEIGHT, DEPTH)
    vis.add_geometry(bbox)

    # Keep track of cone geometries
    cones = [boid_to_cone(b) for b in boids]
    for cone in cones:
        vis.add_geometry(cone)

    for _ in range(steps):
        for b in boids:
            b.separation(boids, radius=8, strength=0.05)
            b.alignment(boids, radius=40, strength=0.05)
            b.cohesion(boids, radius=40, strength=0.0005)
            b.step(dt)

        # Update cones
        for i, b in enumerate(boids):
            vis.remove_geometry(cones[i])  # remove old cone
            cones[i] = boid_to_cone(b)  # create new cone
            vis.add_geometry(cones[i])  # add updated cone

        vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()


boids = [
    Boid3D(
        pos=Vector3D(
            random.uniform(0, WIDTH),
            random.uniform(0, HEIGHT),
            random.uniform(0, DEPTH),
        ),
        vel=Vector3D(
            random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
        ),
        speed=2,
        limit=2,
    )
    for _ in range(100)
]

run_simulation(boids, steps=20000)
