import numpy as np
from panda3d.core import LVecBase3f
import panda3d.core as p3d
from typing import TYPE_CHECKING

from helper_types import Vector3d, Point3d

if TYPE_CHECKING:
    from bhtree import BHTree


class Particle:
    def __init__(self, world=None, index=0, pos=np.zeros(3), mass=1.0, velocity=np.zeros(3)):
        self.index: int = index
        self.world = world
        self.pos: Point3d = pos
        self.next_pos: Point3d = pos.copy()

        self.mass: float = mass
        self.velocity: Vector3d = velocity.copy()
        self.acceleration: Vector3d = np.zeros(3)
        self.force_accum: Vector3d = np.zeros(3)

        # Don't load geometry for unit testing
        if world is not None and not world.no_render:
            self.obj = self.load_object()
        else:
            self.obj = None

    def load_object(self, scale=0.1, transparency=False):
        obj = self.world.loader.loadModel('models/planet_sphere')
        obj.setTexture(self.world.loader.loadTexture('models/moon_1k_tex.jpg'))
        obj.reparentTo(self.world.camera)

        obj.setPos(LVecBase3f(self.pos[0], self.pos[1], self.pos[2]))
        obj.setScale(self.mass * scale)

        obj.setBin('unsorted', 0)
        # obj.setDepthTest(False)

        if transparency:
            obj.setTransparency(p3d.TransparencyAttrib.MAlpha)

        return obj

    def squared_distance(self, other: 'Particle'):
        diff = other.pos - self.pos
        return np.sum(diff ** 2)

    def distance_to(self, other: 'Particle'):
        return np.sqrt(self.squared_distance(other))

    def calculate_force_and_accumulate(self, other: 'Particle' or BHTree, softening=2.0):
        force = self.calculate_force(other, softening=softening)
        # print('Force: ', force)
        self.force_accum += force
        print('Total force:', self.force_accum)

    def calculate_force(self, other: 'Particle' or BHTree, softening=25.0) -> Vector3d:
        direction: np.ndarray[float, float, float] = other.pos - self.pos
        sq_dist = np.sum(direction ** 2)
        magnitude = (self.mass * other.mass) / (sq_dist + softening)
        force = magnitude * direction
        return force

    def reset_forces(self):
        self.force_accum = np.zeros(3)

    def calculate_acceleration(self):
        self.acceleration = self.force_accum / self.mass

    def update_position(self):
        self.pos = self.next_pos.copy()
        if not self.world.no_render:
            self.obj.setPos(LVecBase3f(self.pos[0], self.pos[1], self.pos[2]))
