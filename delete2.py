import numpy as np

def sort_corners(points):
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


def generate_slm_array_spots(aom_bounds, n):

    # pts are in [(x,y), ...]
    # (0,0) (1,0)
    # (0,1) (1,1)

    pts = np.asarray(aom_bounds, dtype=float)

    # sort pts into top left, top right, bottom left, bottom right

    sorted_pts = sort_corners(pts)

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

corners = [[0, 0], [1.0, 0.0], [0.1, 1.0], [0.9, 1.1]]

spot_vectors, spots = generate_slm_array_spots(corners, 2)

print(spots)

# show all these on a plot, checking plotting from top left to bottom right

import matplotlib.pyplot as plt

plt.scatter(spots[:, 0], spots[:, 1], label="spots")
plt.scatter(np.array(corners)[:, 0], np.array(corners)[:, 1], label="corners")

plt.legend()
plt.show()