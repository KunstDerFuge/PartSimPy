import sys

from direct.showbase.ShowBase import ShowBase
import direct.showbase.ShowBaseGlobal
from direct.task.Task import Task
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import LVecBase3f, OrthographicLens

from bhtree import BHTree
from particle import Particle
import direct.gui.DirectGui as DGui
import panda3d.core as p3d
import numpy as np


class World(ShowBase):
    def __init__(self, width=100, height=100, depth=100, particles=None, no_render=False):
        ShowBase.__init__(self)

        if particles is None:
            particles = []

        self.width = width
        self.height = height
        self.depth = depth
        self.particles: list[Particle] = particles
        self.no_render = no_render

        self.title = DGui.OnscreenText(
            text='PartSim: N-body physics simulation',
            parent=self.a2dBottomRight,
            align=p3d.TextNode.A_right,
            style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=.05
        )

        self.calculate_task = taskMgr.add(self.calculate_loop, 'calculate_loop')

        self.setBackgroundColor(0, 0, 0)
        self.disableMouse()

        self.camera.setPos(0, 0, -10)
        # self.camera.look_at(0, 0, 0)
        print('Cameras:', self.camList)
        self.camLens.setFocalLength(0.25)
        self.oobe()

        # lens = OrthographicLens()
        # lens.setFilmSize(40, 30)  # Or whatever is appropriate for your scene
        # base.cam.node().setLens(lens)

        # Keyboard / mouse inputs
        self.accept('escape', sys.exit)

    def initialize_particles(self):
        for p in self.particles:
            p.world = self

    def add_random_particles(self, num):
        for _ in range(num):
            _min = -10
            _max = 10
            x = np.random.uniform(_min, _max)
            y = np.random.uniform(_min, _max)
            z = np.random.uniform(_min, _max)
            mass = np.random.uniform(0.15, 3)
            index = len(self.particles)
            self.add_particle(Particle(self, index, np.array([x, y, z]), mass))

    def add_particle(self, p: Particle):
        self.particles.append(p)

    def calculate_loop(self, task):
        # print('Task time:', task.time)
        # dt = direct.showbase.ShowBaseGlobal.globalClock.getDt()

        t = BHTree(length=100)
        self.calculate_step(t=t, dt=0.1, theta=0.5)
        del t

        return Task.cont

    def calculate_step(self, t: BHTree = None, dt=1.0, theta=1.0):
        # Leapfrog scheme

        # Kick
        for p in self.particles:
            p.velocity += (p.acceleration * (dt / 2.0))
            p.next_pos = p.pos + (p.velocity * dt)

        # Drift
        for p in self.particles:
            p.update_position()

        # Update forces
        if t is not None:
            self.calculate_forces_BH(t, theta)
        else:
            self.calculate_forces()

        # Kick
        for p in self.particles:
            p.calculate_acceleration()
            p.velocity += (p.acceleration * (dt / 2.0))
            p.reset_forces()

    def calculate_forces(self):
        for p1 in self.particles:
            for p2 in self.particles:
                if p1.index == p2.index:
                    continue

                p1.calculate_force_and_accumulate(p2)

    def calculate_forces_BH(self, t: BHTree, theta: float):
        print('Calculating forces...')
        for p in self.particles:
            t.insert(p)

        for p in self.particles:
            t.calculate_force(p, theta)

    def calculate_cg(self):
        pass
