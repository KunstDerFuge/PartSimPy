from typing import Tuple

import numpy as np
import pytest

from bhtree import BHTree
from particle import Particle
from world import World


class TestBHTree:
    @staticmethod
    def create_test_particles(world=None):
        # Positive to negative, in XYZ order, the three dimensions are described by BHTree as
        # EAST - WEST,
        # NORTH - SOUTH,
        # UPPER - LOWER
        return (
            Particle(world=world, pos=np.array([-1, 1,  1])),
            Particle(world=world, pos=np.array([1,  1,  1])),
            Particle(world=world, pos=np.array([-1, -1, 1])),
            Particle(world=world, pos=np.array([1,  -1, 1])),
            Particle(world=world, pos=np.array([-1, 1,  -1])),
            Particle(world=world, pos=np.array([1,  1,  -1])),
            Particle(world=world, pos=np.array([-1, -1, -1])),
            Particle(world=world, pos=np.array([1,  -1, -1]))
        )

    def test_octants(self):
        t = BHTree(np.zeros(3), length=100)
        u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se = self.create_test_particles()

        results = (
            t.which_octant(u_nw),
            t.which_octant(u_ne),
            t.which_octant(u_sw),
            t.which_octant(u_se),
            t.which_octant(l_nw),
            t.which_octant(l_ne),
            t.which_octant(l_sw),
            t.which_octant(l_se)
        )

        assert results == (
            BHTree.U_NW, BHTree.U_NE, BHTree.U_SW, BHTree.U_SE, BHTree.L_NW, BHTree.L_NE, BHTree.L_SW, BHTree.L_SE
        )

    def test_containment(self):
        t = BHTree(np.zeros(3), length=100)
        u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se = self.create_test_particles()
        for p in (u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se):
            t.insert(p)

        print(t.subnodes.values())
        assert all(t.subnodes.values())

        # Baseline, main tree contains all particles
        results_1 = (
            t.is_containing(u_nw),
            t.is_containing(u_ne),
            t.is_containing(u_sw),
            t.is_containing(u_se),
            t.is_containing(l_nw),
            t.is_containing(l_ne),
            t.is_containing(l_sw),
            t.is_containing(l_se),
        )
        print(results_1)
        assert all(results_1)

        # Test that each subnode contains its corresponding particle
        results_2 = (
            t.subnodes[BHTree.U_NW].is_containing(u_nw),
            t.subnodes[BHTree.U_NE].is_containing(u_ne),
            t.subnodes[BHTree.U_SW].is_containing(u_sw),
            t.subnodes[BHTree.U_SE].is_containing(u_se),
            t.subnodes[BHTree.L_NW].is_containing(l_nw),
            t.subnodes[BHTree.L_NE].is_containing(l_ne),
            t.subnodes[BHTree.L_SW].is_containing(l_sw),
            t.subnodes[BHTree.L_SE].is_containing(l_se),
        )
        print(results_2)
        assert all(results_2)

        # Test that no other octants contain a particle incorrectly
        results_3 = (
            t.subnodes[BHTree.U_NE].is_containing(u_nw),
            t.subnodes[BHTree.U_NW].is_containing(u_ne),
            t.subnodes[BHTree.U_NW].is_containing(u_sw),
            t.subnodes[BHTree.U_NE].is_containing(u_se),
            t.subnodes[BHTree.U_NW].is_containing(l_nw),
            t.subnodes[BHTree.U_NE].is_containing(l_ne),
        )
        print(results_3)
        assert not any(results_3)

    def test_mass_accumulation(self):
        t = BHTree(np.zeros(3), length=10)
        u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se = self.create_test_particles()
        for p in (u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se):
            t.insert(p)

        assert t.num_particles == 8
        assert t.mass == 8.0

    def test_cg_accumulation(self):
        t = BHTree(np.zeros(3), length=10)
        u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se = self.create_test_particles()
        for p in (u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se):
            t.insert(p)

        print('Accumulated CG:', t.pos)
        assert t.pos == pytest.approx(np.zeros(3))

    def test_acceleration(self):
        u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se = self.create_test_particles()
        w = World(particles=[u_nw, u_ne, u_sw, u_se, l_nw, l_ne, l_sw, l_se], no_render=True)
        w.initialize_particles()

        print(w.particles)

        for _ in range(20):
            t = BHTree(length=10)
            w.calculate_step(t=t, dt=0.1, theta=0.5)
            del t

        print('U_NW velocity:', u_nw.velocity)
        print('U_NW acceleration:', u_nw.acceleration)
        print('U_NW forces:', u_nw.force_accum)

        assert all((u_nw.velocity[0] > 0, u_nw.velocity[1] > 0, u_nw.velocity[2] > 0))


