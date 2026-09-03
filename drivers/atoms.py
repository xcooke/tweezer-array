
from numpy import random

from drivers.tweezer_detection import detect_clumps, get_theoretical_slm_tweezer_locations

class Atoms():

    def __init__(self, camera_tweezers):
        self.camera_tweezers = camera_tweezers

        # atom positions should be stored in a list of tuples, where each tuple is (x, y) in pixels
        # initially no atoms are loaded so 

        self.locations = []

    def load_atoms(self, threshold=500):

        # reset the atoms list
        self.locations = []

        # check camera_tweezers for existing tweezers

        #tweezer_image = self.camera_tweezers.get_image()

        # extract tweezer locations from camera_tweezers
        # look for spots with pixel value above 500

        #tweezer_locations = detect_clumps(tweezer_image, threshold=threshold)

        tweezer_locations = get_theoretical_slm_tweezer_locations()

        # now iterate through tweezer_locations and with probability 0.5, add an atom to self.atoms at that location

        for loc in tweezer_locations:
            if random.random() < 0.5:
                self.locations.append(loc)

        print(f"Loaded {len(self.locations)} atoms out of {len(tweezer_locations)} tweezers.")

    def move_atom(self, src, dst):

        # Compare coordinates explicitly because self.locations
        # may contain NumPy arrays.

        src_index = None
        dst_occupied = False

        for i, loc in enumerate(self.locations):
            if loc[0] == src[0] and loc[1] == src[1]:
                src_index = i

            if loc[0] == dst[0] and loc[1] == dst[1]:
                dst_occupied = True

        if src_index is not None and not dst_occupied:
            self.locations.pop(src_index)
            self.locations.append(dst)
            print(f"Moved atom from {src} to {dst}")
        else:
            print(
                f"Cannot move atom from {src} to {dst}, "
                "either src is not occupied or dst is already occupied."
            )