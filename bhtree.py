from typing import Final, Dict, Optional

from particle import Particle
import numpy as np

from helper_types import Point3d


class BHTree:
    U_NW: Final = 0
    U_NE: Final = 1
    U_SW: Final = 2
    U_SE: Final = 3
    L_NW: Final = 4
    L_NE: Final = 5
    L_SW: Final = 6
    L_SE: Final = 7

    def __init__(self, center=np.zeros(3), length=0.0, parent: 'BHTree' = None, octant=0):
        self.subnodes: Dict[int, Optional['BHTree']] = {
            BHTree.U_NW: None,
            BHTree.U_NE: None,
            BHTree.U_SW: None,
            BHTree.U_SE: None,
            BHTree.L_NW: None,
            BHTree.L_NE: None,
            BHTree.L_SW: None,
            BHTree.L_SE: None
        }
        self.body: Optional[Particle] = None
        self.num_particles: int = 0
        self.pos: Point3d = np.zeros(3)
        self.mass: float = 0.0

        if parent is None:
            self.center = center
            self.length = length

        else:
            # Get vector to new center based on node length and
            # this octant's position relative to its parent
            vector_to_center = np.zeros(3)
            self.length = parent.length / 2.0
            magnitude = self.length / 2.0

            if octant == BHTree.U_NW:
                vector_to_center = np.array([-1, 1,  1]) * magnitude
            elif octant == BHTree.U_NE:
                vector_to_center = np.array([1,  1,  1]) * magnitude
            elif octant == BHTree.U_SW:
                vector_to_center = np.array([-1, -1, 1]) * magnitude
            elif octant == BHTree.U_SE:
                vector_to_center = np.array([1,  -1, 1]) * magnitude
            elif octant == BHTree.L_NW:
                vector_to_center = np.array([-1, 1,  -1]) * magnitude
            elif octant == BHTree.L_NE:
                vector_to_center = np.array([1,  1,  -1]) * magnitude
            elif octant == BHTree.L_SW:
                vector_to_center = np.array([-1, -1, -1]) * magnitude
            elif octant == BHTree.L_SE:
                vector_to_center = np.array([1,  -1, -1]) * magnitude

            self.center = parent.center + vector_to_center

    def is_containing(self, p: Particle) -> bool:
        """
        Return whether the particle's position is within this octant
        """
        distance_to_side = self.length / 2.0
        top = self.center + np.array([0, 0, 1]) * distance_to_side
        bottom = self.center + np.array([0, 0, -1]) * distance_to_side
        left = self.center + np.array([-1, 0, 0]) * distance_to_side
        right = self.center + np.array([1, 0, 0]) * distance_to_side
        front = self.center + np.array([0, -1, 0]) * distance_to_side
        back = self.center + np.array([0, 1, 0]) * distance_to_side

        p_x = p.pos[0]
        p_y = p.pos[1]
        p_z = p.pos[2]

        if left[0]   <= p_x <= right[0] and \
           front[1]  <= p_y <= back[1] and \
           bottom[2] <= p_z <= top[2]:
            return True
        return False

    def is_internal(self) -> bool:
        """
        Return whether this node is internal (Not a leaf node)
        """
        return any(self.subnodes.values())

    def is_empty(self) -> bool:
        """
        Return whether this node recursively contains a particle or not
        """
        # Note: self.body will point to the first particle inserted into it even if subdivided
        return self.num_particles == 0

    def is_octant_unallocated(self, octant: int) -> bool:
        return self.subnodes[octant] is None

    def which_octant(self, p: Particle) -> int:
        """
        Return which octant a particle should belong to
        """
        p_x = p.pos[0]
        p_y = p.pos[1]
        p_z = p.pos[2]

        t_x = self.center[0]
        t_y = self.center[1]
        t_z = self.center[2]

        if p_z < t_z:           # Particle is below center
            if p_y < t_y:       # Particle is south of center
                if p_x < t_x:   # Particle is west of center
                    return BHTree.L_SW
                else:           # Particle is east of center
                    return BHTree.L_SE

            else:               # Particle is north of center
                if p_x < t_x:   # Particle is west of center
                    return BHTree.L_NW
                else:           # Particle is east of center
                    return BHTree.L_NE

        else:                   # Particle is above center
            if p_y < t_y:       # Particle is south of center
                if p_x < t_x:   # Particle is west of center
                    return BHTree.U_SW
                else:           # Particle is east of center
                    return BHTree.U_SE

            else:               # Particle is north of center
                if p_x < t_x:   # Particle is west of center
                    return BHTree.U_NW
                else:           # Particle is east of center
                    return BHTree.U_NE

    def insert(self, p: Particle) -> None:
        """
        Insert a particle into this octant
        """
        if not self.is_containing(p):
            # Don't insert this particle if it isn't in this octant
            # print('Octant at', self.pos, 'does NOT contain particle at', p.pos)
            print('WARNING: particle inserted into octant not containing it')
            print('Particle pos:', p.pos)
            print('Octant pos:', self.pos, 'length', self.length)
            return

        if self.is_empty():
            # Insert the particle into this node and take its CG
            self.body = p
            self.pos = p.pos.copy()
            self.mass = p.mass
            # print('Set octant\'s particle pointer.')
        else:
            # Node already contains a body
            # Determine which subnode octant we will insert into
            octant = self.which_octant(p)

            if self.is_internal():
                # Contains subnodes, so insert into one of them
                self.add_to_CG(p)
                if self.is_octant_unallocated(octant):
                    self.subnodes[octant] = BHTree(parent=self, octant=octant)

                # print('Octant is full, inserting into subnode...')
                self.subnodes[octant].insert(p)

            else:
                # Node contains a particle and is external (contains no subnodes)
                # Subdivide self into subnodes and insert existing particle into deeper subnode
                existing_particle_octant = self.which_octant(self.body)
                if self.is_octant_unallocated(existing_particle_octant):
                    self.subnodes[existing_particle_octant] = BHTree(parent=self, octant=existing_particle_octant)

                # print('Subdividing self...')
                self.subnodes[existing_particle_octant].insert(self.body)
                self.body = None

                # Finally, insert the new particle into a deeper subnode
                if self.is_octant_unallocated(octant):
                    self.subnodes[octant] = BHTree(parent=self, octant=octant)

                # print('Inserting into deeper subnode...')
                self.subnodes[octant].insert(p)
                self.add_to_CG(p)

        self.num_particles += 1

    def add_to_CG(self, p: Particle) -> None:
        """
        Update the CG of this node
        """
        # Calculate the center of mass of the existing accumulated CG and the new particle
        cg_pos = (p.pos * p.mass + self.pos * self.mass) / (p.mass + self.mass)
        # p_x = p.pos[0]
        # p_y = p.pos[1]
        # p_z = p.pos[2]
        #
        # t_x = self.pos[0]
        # t_y = self.pos[1]
        # t_z = self.pos[2]
        #
        # x_cg = (t_x * self.mass + p_x * p.mass) / self.mass
        # y_cg = (t_y * self.mass + p_y * p.mass) / self.mass
        # z_cg = (t_z * self.mass + p_z * p.mass) / self.mass
        #
        # self.pos = np.array([x_cg, y_cg, z_cg])
        self.pos = cg_pos
        self.mass += p.mass
        # print('Octant centered at', self.center, 'with', self.num_particles, 'parts', 'length', self.length, 'CG now', self.pos, 'total mass', self.mass)

    def distance_to(self, p: Particle) -> float:
        return np.linalg.norm(p.pos - self.pos)

    def calculate_force(self, p: Particle, theta) -> None:
        """
        Calculate and the force exerted on particle p and update it
        """
        if not self.is_internal() and not self.is_empty() and self.body.index is not p.index:
            # This node contains only one body; its CG is exactly equal to the contained particle
            # print('Node is leaf...')
            p.calculate_force_and_accumulate(self.body)

        else:
            # Node is internal
            distance = self.distance_to(p)
            if distance == 0:
                return

            if (self.length / distance) < theta:
                # Node is far enough away; approximate force exerted on particle p by using node's CG
                # print('Node is distant...')
                p.calculate_force_and_accumulate(self)

            else:
                # Particle is close to this node; calculate forces exerted on particle p from all contained particles
                for octant, subnode in self.subnodes.items():
                    if subnode is not None:
                        # print('Node is nearby, recursing...')
                        subnode.calculate_force(p, theta)
