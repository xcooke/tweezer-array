import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import os
from math import dist

from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography import analysis

from drivers.tweezer_detection import detect_clumps
from drivers.rearrangement import solve_rearrangement, apply_moves
from drivers.tetris_algorithm import tetris_moves, _apply_batch
from drivers.tetris_plot import plot_tetris_states
from drivers.trajectory import calculate_trajectory
from drivers.batches_to_states import convert_batches_to_states

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

        phase_path = f"{self.phase_folder_path}\\phase.pkl"

        print(phase_path)

        phase_array = np.load(phase_path, allow_pickle=True)

        self.slm.set_phase(phase_array)

        print("SLM phase set")

    def clear_phase(self):

        self.slm.set_phase(np.zeros((1200, 1920)))

        print("SLM phase cleared")


    def sort_corners(self, points):
        # points = [(x, y), (x, y), (x, y), (x, y)]

        # require that two smallest y values are the top row
        # so can't be rotated too much

        # Assuming image coordinates: smaller y = higher/top
        points = sorted(points, key=lambda p: p[1])

        top = sorted(points[:2], key=lambda p: p[0])
        bottom = sorted(points[2:], key=lambda p: p[0])

        return {
            "top_left": top[0],
            "top_right": top[1],
            "bottom_left": bottom[0],
            "bottom_right": bottom[1],
        }


    def plot_target_array(self, corners, spots):

        plt.scatter(spots[:, 0], spots[:, 1], label="spots")
        plt.scatter(np.array(corners)[:, 0], np.array(corners)[:, 1], label="corners")

        plt.legend()
        plt.savefig(f"target_arrays/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")


    def generate_slm_array_spots(self, aom_bounds, n):

        # pts are in [(x,y), ...]
        # (0,0) (1,0)
        # (0,1) (1,1)

        pts = np.asarray(aom_bounds, dtype=float) # do I need this?
        print(f"aom_bounds are: {aom_bounds}")
        print(f"converted into numpy array: {pts}")

        # sort pts into top left, top right, bottom left, bottom right

        sorted_pts = self.sort_corners(pts)

        print(sorted_pts)

        left_x_max = max(sorted_pts["top_left"][0], sorted_pts["bottom_left"][0])
        right_x_min = min(sorted_pts["top_right"][0], sorted_pts["bottom_right"][0])
        top_y_max = max(sorted_pts["top_left"][1], sorted_pts["top_right"][1])
        bottom_y_min = min(sorted_pts["bottom_left"][1], sorted_pts["bottom_right"][1])

        # find x side length

        x_side_length = right_x_min - left_x_max
        y_side_length = bottom_y_min - top_y_max

        # find minimum side length to make square
        side = min(x_side_length, y_side_length)

        # find centre of quadrilateral, hopefully this should work!

        cx = (left_x_max + right_x_min) / 2
        cy = (top_y_max + bottom_y_min) / 2

        # Axis-aligned square centred on quadrilateral
        x = np.linspace(cx - side / 2, cx + side / 2, n)
        y = np.linspace(cy - side / 2, cy + side / 2, n)

        X, Y = np.meshgrid(x, y)

        spots = np.column_stack((X.ravel(), Y.ravel()))
        spot_vectors = spots.T

        return spot_vectors, spots


    def save_image_stack(self, images, filename, type):
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = np.atleast_1d(axes).ravel()

        vmin = np.nanmin(images)
        vmax = np.nanmax(images)

        for ax, img in zip(axes, images):
            im = ax.imshow(img, vmin=vmin, vmax=vmax)
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
        plt.title(f"Subimages from camera after {type} optimisation")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


    def compare_spot_uniformity(self, hologram, spot_vectors, folder, optimisation_type):

        self.fs.slm.set_phase(hologram.get_phase(), settle=True)                 # Write hologram.
        self.fs.cam.flush()
        img = self.fs.cam.get_image()                                        # Grab image.

        plt.imsave(f"{folder}/after_{optimisation_type}_optimisation_camera.png", img)
        plt.close()

        subimages = analysis.take(img, vectors=spot_vectors, size=4)

        self.save_image_stack(subimages, f"{folder}/after_{optimisation_type}_optimisation_subimages.png", optimisation_type)

        powers = analysis.image_normalization(subimages)

        powers_norm = powers / np.mean(powers)
        powers_norm_std_gs = np.std(powers_norm)

        plt.hist(powers_norm)
        plt.title("GS Powers (std={:.2f}%)".format(powers_norm_std_gs * 100))
        plt.savefig(f"{folder}/after_{optimisation_type}_optimisation_histogram.png", dpi=300, bbox_inches="tight")
        plt.close()

    def generate_slm_phase_pattern(self, aom_bounds, n, hologram_size=2048):

        print(aom_bounds)

        # aom_bounds is a quadrilateral
        # with corners at integer values [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        # want to generate an array of n x n spots within the aom_bounds quadrilateral

        spot_vectors, spots = self.generate_slm_array_spots(aom_bounds, n)

        # write spots to a pickle file for later use

        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        folder = f"slm_spots/{n}x{n}_{dt}"
        os.makedirs(folder, exist_ok=True)

        spots_filename = f"{folder}/spots.pkl"
        phase_filename = f"{folder}/phase.pkl"

        with open(spots_filename, "wb") as f:
            pickle.dump(spots, f)

        print(f"spot_vectors_ij: {spot_vectors}")

        spot_vectors_knm = toolbox.convert_vector(
            spot_vectors,
            from_units="ij",
            to_units="knm",
            hardware=self.fs,
        )

        print(f"spot_vectors_knm: {spot_vectors_knm}")

        hologram = SpotHologram(shape=(hologram_size, hologram_size), spot_vectors=spot_vectors, basis='ij', cameraslm=self.fs)

        # optimise computationally
        hologram.optimize(
            'WGS-Kim', 
            feedback='computational_spot', 
            stat_groups=['computational_spot'], 
            maxiter=50
        )

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        hologram.plot_farfield(
            axs=axs,
            title="Farfield Computational Optimization",
        )

        # Save using the figure YOU created
        fig.tight_layout()
        fig.savefig(f"{folder}/farfield_computational_optimisation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


        self.compare_spot_uniformity(hologram, spot_vectors, folder, "computational")


        ## now optimise experimentally
        """
        hologram.spot_integration_width_ij = 7
        hologram.optimize(
            'WGS-Kim',
            maxiter=50,
            feedback='experimental_spot',
            stat_groups=['computational_spot', 'experimental_spot'],
            fixed_phase=False
        )

        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        hologram.plot_farfield(
            axs=axs,
            title="Farfield Experimental Optimization (Not Camera)",
        )

        # Save using the figure YOU created
        fig.tight_layout()
        fig.savefig(f"{folder}/farfield_experimental_optimisation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


        self.compare_spot_uniformity(hologram, spot_vectors, folder, "experimental")

        ax = hologram.plot_stats(show=False)
        fig = ax.get_figure()
        fig.savefig(f"{folder}/optimisation_stats.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        """

        phase = hologram.get_phase()

        with open(phase_filename, "wb") as f:
            pickle.dump(phase, f)

        self.slm.set_phase(phase)

        return folder


    def fourier_calibrate(self):

        print("Beginning Fourier calibration")

        self.cam.set_exposure(6e-4) # this might need changing...

        
        self.fs.fourier_calibrate(
            array_center=(480,0), #(500,20)
            array_shape=8,                 # Size of the calibration grid (Nx, Ny) [#]
            array_pitch=100, #16                 # Pitch of the calibration grid (x, y) [knm]
            #plot=2,
        )

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
        if np.max(image) < 100:
            raise ValueError(f"Image max value is {np.max(image)} < 100, not expecting this")

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
        with open(f"{self.phase_folder_path}\\spots.pkl", "rb") as f:
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

            # Get the locations belonging to this row
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

        # plot each region in array_size x array_size grid

        self.plot_regions(fluorescence_image_regions)

        # iterate over regions and if max > some threshold value then treat as filled
        # mark each tweezer_location as filled or empty

        

        fluorescence_threshold = 200

        filled = {}

        number_filled_sites = 0

        for site, image in fluorescence_image_regions.items():
            row, column = site
            if np.max(image) > fluorescence_threshold:
                filled[(row, column)] = 1
                number_filled_sites += 1
            else:
                filled[(row, column)] = 0

        return filled, sites, array_size, number_filled_sites

    def generate_rearrange_movements(self):

        filled, sites, array_size, number_filled_sites = self.identify_filled_atom_sites()

        # convert filled into a 2D array

        initial = np.zeros((array_size, array_size), dtype=int)
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


    def generate_tetris_rearrange_movements(self, T, dt):

        filled, sites, array_size, number_filled_sites = self.identify_filled_atom_sites()

        # convert filled into a 2D array

        initial = np.zeros((array_size, array_size), dtype=int)
        for site, value in filled.items():
            row, column = site
            initial[row, column] = value

        print("Initial:")
        print(initial)

        # round down sqrt of number_filled_sites to get target size

        target_size = int(np.floor(np.sqrt(number_filled_sites)))

        # target array is size array_size x array_size of 0s
        # with a target_size x target_size square of 1s in the centre

        target = np.zeros((array_size, array_size), dtype=int)
        start_row = (array_size - target_size) // 2
        start_col = (array_size - target_size) // 2
        target[start_row:start_row + target_size, start_col:start_col + target_size] = 1

        print("Target:")
        print(target)

        batches, final_state, extra_atoms = tetris_moves(
            initial,
            target,
            5,
        )

        print(f"Initial atoms: {initial.sum()}")
        print(f"Target atoms:  {target.sum()}")
        print(f"Extra atoms:   {len(extra_atoms)}")
        print(f"Move batches:  {len(batches)}")
    
        for batch in batches:
            print(batch)

        
        # ============================================================
        # Replay moves so that we can save every intermediate state.
        # ============================================================

        states = [initial.copy()]

        current_state = initial.copy()

        for batch in batches:

            current_state = _apply_batch(
                current_state,
                batch["moves"],
            )

            states.append(
                current_state.copy()
            )

        plot_tetris_states(
            states,
            target,
            batches,
        )

        states = convert_batches_to_states(
            batches,
            sites,
            T,
            dt,
            "characterisations\\2026-09-09_17-39-02\\position_calibration.pkl",
        )

        return states

    def close(self):
    
        self.slm.close()

        print("SLM closed")
        