
from scipy import ndimage
import numpy as np
import pickle

def detect_clumps(arr, threshold=500):

    # output centres in format [(x1, y1), (x2, y2), ...] where x is column index and y is row index

    # arr = your 2D numpy array
    mask = arr > threshold

    # Label connected clumps
    labels, n_clumps = ndimage.label(mask)

    # Centre of each clump
    centres = ndimage.center_of_mass(
        mask,
        labels,
        range(1, n_clumps + 1)
    )

    # Convert from NumPy's (row, column) order to the public (x, y) format.
    return [(float(column), float(row)) for row, column in centres]


def get_theoretical_slm_tweezer_locations(phase_folder_path):

    # read out the theoretical SLM tweezer locations from the pickle file
    with open(f"{phase_folder_path}\\spots.pkl", "rb") as f:
        spots = pickle.load(f)

    # sort spots by y and then x

    spots = spots[np.lexsort((spots[:, 0], spots[:, 1]))]

    return spots
