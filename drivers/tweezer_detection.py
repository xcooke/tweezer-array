
from scipy import ndimage
import numpy as np
import pickle

def detect_clumps(arr, threshold=500):

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

    print(centres)

    # and round the centres to the nearest integer pixel coordinates

    #centres = [(int(round(c[1])), int(round(c[0]))) for c in centres]
    # dont round

    return centres


def get_theoretical_slm_tweezer_locations():

    # read out the theoretical SLM tweezer locations from the pickle file
    with open("4x4_spot_array_spots.pkl", "rb") as f:
        spots = pickle.load(f)

    # sort spots by y and then x

    spots = spots[np.lexsort((spots[:, 0], spots[:, 1]))]

    return spots
