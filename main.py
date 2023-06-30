import numpy as np

from particle import Particle
from world import World

from direct.showbase.ShowBase import ShowBase


if __name__ == '__main__':
    # wp.init()
    world = World()
    world.add_random_particles(300)
    world.run()
