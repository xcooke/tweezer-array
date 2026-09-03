import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import os
from math import dist

from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import SpotHologram

from drivers.tweezer_detection import detect_clumps
from drivers.rearrangement import solve_rearrangement, apply_moves


class SLMCam:
    def __init__(self, cam, slm):

        print("\nStarting SLMCam initialisation\n")
        self.cam = cam
        self.slm = slm

        self.fs = FourierSLM(self.cam, self.slm)
        self.fs.load_calibration("fourier")
        

        print("\nFourierSLM initialised\n")

    def set_phase(self):
    
        # read 4x4_spot_array_phase.pkl

        phase_array = np.load("4x4_spot_array_phase.pkl", allow_pickle=True)

        self.slm.set_phase(phase_array)

        print("SLM phase set")

    def clear_phase(self):

        self.slm.set_phase(np.zeros((1200, 1920)))

        print("SLM phase cleared")

    def generate_slm_array_spots(self, aom_bounds, n):
        """
        Generate an n x n grid inside a convex quadrilateral.

        aom_bounds:
            Four (x, y) corners in ANY order.

        n:
            Number of spots along each dimension.

        Returns:
            spots: [[x1, y1], [x2, y2], ...]
            spot_vectors: [[x1, x2, ...], [y1, y2, ...]]
        """

        pts = np.asarray(aom_bounds, dtype=float)

        if pts.shape != (4, 2):
            raise ValueError("aom_bounds must contain exactly four (x, y) points")

        # ------------------------------------------------------------
        # 1. Put corners into cyclic order around the quadrilateral
        # ------------------------------------------------------------

        center = pts.mean(axis=0)

        angles = np.arctan2(
            pts[:, 1] - center[1],
            pts[:, 0] - center[0]
        )

        pts = pts[np.argsort(angles)]

        # ------------------------------------------------------------
        # 2. Pick a deterministic starting corner
        #
        # Choose the upper-most point (smallest y), breaking ties
        # using smallest x.
        # ------------------------------------------------------------

        start = np.lexsort((pts[:, 0], pts[:, 1]))[0]
        pts = np.roll(pts, -start, axis=0)

        p0, p1, p2, p3 = pts

        # ------------------------------------------------------------
        # 3. Bilinear interpolation
        # ------------------------------------------------------------

        u = np.linspace(0.0, 1.0, n)
        v = np.linspace(0.0, 1.0, n)

        U, V = np.meshgrid(u, v)

        spots = (
            (1 - U)[..., None] * (1 - V)[..., None] * p0
            + U[..., None]     * (1 - V)[..., None] * p1
            + U[..., None]     * V[..., None]       * p2
            + (1 - U)[..., None] * V[..., None]     * p3
        )

        # SLM/AOM coordinates must be integer-valued.
        #spots = np.rint(spots).astype(int)
        # dont round

        spots = spots.reshape(-1, 2)

        spot_vectors = spots.T

        return spot_vectors, spots


    def generate_slm_phase_pattern(self, aom_bounds, n):

        print(aom_bounds)

        # aom_bounds is a quadrilateral
        # with corners at integer values [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        # want to generate an array of n x n spots within the aom_bounds quadrilateral

        spot_vectors, spots = self.generate_slm_array_spots(aom_bounds, n)

        # write spots to a pickle file for later use

        with open("4x4_spot_array_spots.pkl", "wb") as f:
            pickle.dump(spots, f)

        hologram = SpotHologram(shape=(2048, 2048), spot_vectors=spot_vectors, basis='ij', cameraslm=self.fs)

        hologram.optimize('WGS-Kim', feedback='computational_spot', stat_groups=['computational_spot'], maxiter=50)

        phase = hologram.get_phase()

        with open("4x4_spot_array_phase.pkl", "wb") as f:
            pickle.dump(phase, f)

        self.slm.set_phase(phase)


    def fourier_calibrate(self):

        print("Beginning Fourier calibration")

        self.cam.set_exposure(6e-4) # this might need changing...

        self.fs.fourier_calibrate(
            array_center=(500,20),
            array_shape=8,                 # Size of the calibration grid (Nx, Ny) [#]
            array_pitch=16,                 # Pitch of the calibration grid (x, y) [knm]
            #plot=2
        );

        self.fs.save_calibration("fourier")

        print("Fourier calibration complete")


    def create_new_characterisation_file(self):
        # create a folder named with the date and time of now to store the characterisation data,
        # inside characterisations/

        #write date and time into a string
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        characterisation_folder_path = f"characterisations/{dt_string}"
        self.characterisation_folder_path = characterisation_folder_path

        # create new folder
        os.mkdir(characterisation_folder_path)

        characterisation_file_path = f"{characterisation_folder_path}/raw.pkl"
        self.characterisation_file_path = characterisation_file_path

        # create a new pickle file to store the characterisation data
        with open(characterisation_file_path, "wb") as f:
            pickle.dump({}, f)

        raw_images_folder_path = f"{self.characterisation_folder_path}/raw_images"
        self.raw_images_folder_path = raw_images_folder_path
        if not os.path.exists(raw_images_folder_path):
            os.mkdir(raw_images_folder_path)

    def save_characterisation_image(self, x_freq, y_freq):

        print(x_freq, y_freq)

        image = self.cam.get_image()

        print("got image")

        # check if max value is above 800

        if np.max(image) > 1021:
            raise ValueError(f"Image max value is {np.max(image)} > 1021, not expecting this")
        if np.max(image) < 200:
            raise ValueError(f"Image max value is {np.max(image)} < 200, not expecting this")

        # write images to a file as an object

        with open(self.characterisation_file_path, "rb") as f:
            data = pickle.load(f)

        data[(x_freq, y_freq)] = image

        with open(self.characterisation_file_path, "wb") as f:
            pickle.dump(data, f)

        
        plt.imsave(f"{self.raw_images_folder_path}/{x_freq}_{y_freq}.png", image, cmap="gray")


    def get_tweezer_locations(self, threshold=500):

        tweezer_image = self.cam.get_image()
        print("max value in tweezer image:", np.max(tweezer_image))
        tweezer_locations = detect_clumps(tweezer_image, threshold=threshold)

        return tweezer_locations

    def get_theoretical_slm_tweezer_locations(self):

        # read out the theoretical SLM tweezer locations from the pickle file
        with open("4x4_spot_array_spots.pkl", "rb") as f:
            spots = pickle.load(f)

        # sort spots by y and then x

        spots = spots[np.lexsort((spots[:, 0], spots[:, 1]))]

        return spots

    def plot_regions(self, regions):

        num_rows = int(np.sqrt(len(regions)))

        fig, axs = plt.subplots(num_rows, num_rows, figsize=(8, 8))

        for site, image in regions.items():
            row, column = site

            axs[row, column].imshow(image, cmap='gray')
            axs[row, column].set_title(f"Region ({row},{column})")

        plt.tight_layout()
        # save plot to folder regions/ with date
        plt.savefig(f"regions/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")


    def identify_filled_atom_sites(self):

        print("Identifying filled atom sites")

        fluorescence_image = self.camera_fluorescence.get_image()

        tweezer_locations = self.get_theoretical_slm_tweezer_locations()

        print(f"There are {len(tweezer_locations)} tweezer locations")

        if len(tweezer_locations) != 16:
            raise ValueError(f"There are not 16 tweezer locations, there are {len(tweezer_locations)}")

        # check if tweezer_locations is a square number

        if int(np.sqrt(len(tweezer_locations)))**2 != len(tweezer_locations):
            raise ValueError(f"There are not a square number of tweezer locations, there are {len(tweezer_locations)}")


        array_size = int(np.sqrt(len(tweezer_locations)))

        # Sort all tweezer locations from smallest y to largest y
        locations_sorted_by_y = sorted(
            tweezer_locations,
            key=lambda location: location[1]
        )

        sites = {}

        for row in range(array_size):

            # Get the 4 locations belonging to this row
            row_locations = locations_sorted_by_y[
                row * array_size:(row + 1) * array_size
            ]

            # Sort this row from smallest x to largest x
            row_locations = sorted(
                row_locations,
                key=lambda location: location[0]
            )

            # Assign columns
            for column, location in enumerate(row_locations):
                sites[(row, column)] = location


        # get minimum distance between tweezer locations

        print("Finding tweezer separation")
        
        tweezer_separation = min(
            dist(tweezer_locations[i], tweezer_locations[j])
            for i in range(len(tweezer_locations))
            for j in range(i + 1, len(tweezer_locations))
        )

        print(f"Smallest tweezer separation: {tweezer_separation}")

        # cut fuorescence_image into squares with centres at tweezer_loctions and with half side width 10 pixels

        half_side_width = int(round((tweezer_separation * 0.8) * 0.5))


        fluorescence_image_regions = {}

        for site in sites:
            row, column = site
            x, y = sites[site]

            x = int(round(x))
            y = int(round(y))

            region = fluorescence_image[y-half_side_width:y+half_side_width, x-half_side_width:x+half_side_width]

            fluorescence_image_regions[(row, column)] = region

        # plot each region in a 4x4 grid

        self.plot_regions(fluorescence_image_regions)

        # iterate over regions and if max > some threshold value then treat as filled
        # mark each tweezer_location as filled or empty

        

        fluorescence_threshold = 200

        filled = {}

        for site, image in fluorescence_image_regions.items():
            row, column = site
            if np.max(image) > fluorescence_threshold:
                filled[(row, column)] = 1
            else:
                filled[(row, column)] = 0

        return filled, sites

    def generate_rearrange_movements(self):

        filled, sites = self.identify_filled_atom_sites()

        # convert filled into a 2D array

        initial = np.zeros((4, 4), dtype=int)
        for site, value in filled.items():
            row, column = site
            initial[row, column] = value

        print(initial)

        moves, target = solve_rearrangement(initial)

        print("Initial:")
        print(initial)

        print("\nTarget:")
        print(target.astype(int))

        print("\nNumber of moves:", len(moves))

        print("\nMoves:")
        for i, (src, dst) in enumerate(moves, 1):
            print(f"{i:3}: {src} -> {dst}: {sites[src]} -> {sites[dst]}")

        
        final = apply_moves(initial, moves)
        #print("\nFinal:")
        #print(final)
        assert np.array_equal(final, target)

        return moves, sites 




    def close(self):
    
        self.slm.close()

        print("SLM closed")
        