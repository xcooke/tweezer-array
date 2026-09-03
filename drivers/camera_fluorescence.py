#!/usr/bin/env python3

import numpy as np

from drivers import atoms


class CameraFluorescence():

    def __init__(self, camera_hardware, atoms):
        self.camera_hardware = camera_hardware
        self.atoms = atoms

        self.shape = self.camera_hardware.shape

    def set_woi(self):
        self.shape = self.camera_hardware.shape

    def get_image(self):

        # make empty image

        image = np.zeros(self.shape, dtype=np.uint16)

        # get atom positions from atoms

        atom_positions = self.atoms.locations

        for pos in atom_positions:
            x, y = pos

            x = int(round(x))
            y = int(round(y))
            
            # add a bright spot at the atom position
            image[y, x] = 500

        return image

