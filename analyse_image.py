import pickle
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import TwoSlopeNorm
import os
import time

folder_path = 'characterisations\\2026-09-09_17-39-02'


# read pickle file
def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def gaussian_2d(coords, amplitude, xo, yo, sigma):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    sigma = float(sigma)

    g = amplitude * np.exp(
        -(((x - xo)**2 + (y - yo)**2) / (2 * sigma**2))
    )
    return g.ravel()


def fit_gaussian_2d(data):
    data = np.asarray(data)
    ny, nx = data.shape

    # maximum value in image
    max_value = np.max(data)
    if max_value > 1000:
        print(f"Warning: maximum value in image is {max_value} > 1000, not expecting this")
        quit()

    y = np.arange(ny)
    x = np.arange(nx)
    x, y = np.meshgrid(x, y)

    # Initial guess from data moments
    offset = np.median(data)

    if offset >= 20:
        print(f"Warning: offset is {offset} >= 20, not expecting this")
        quit()

    data = data - offset

    amplitude0 = data.max()

    total = np.sum(data)
    if total <= 0:
        total = np.sum(data)

    #xo0 = np.sum(x * data) / np.sum(data)
    #yo0 = np.sum(y * data) / np.sum(data)

    # set xo0 to maximum value in image
    max_index = np.unravel_index(np.argmax(data), data.shape)
    yo0, xo0 = max_index

    """
    col = data[:, int(np.clip(round(xo0), 0, nx - 1))]
    row = data[int(np.clip(round(yo0), 0, ny - 1)), :]

    sigma_x0 = np.sqrt(np.abs(np.sum(((np.arange(nx) - xo0) ** 2) * row) / np.sum(row)))
    sigma_y0 = np.sqrt(np.abs(np.sum(((np.arange(ny) - yo0) ** 2) * col) / np.sum(col)))

    # Fallbacks if moments fail
    if not np.isfinite(sigma_x0) or sigma_x0 == 0:
        sigma_x0 = nx / 4
    if not np.isfinite(sigma_y0) or sigma_y0 == 0:
        sigma_y0 = ny / 4

    sigma0 = np.sqrt((sigma_x0**2 + sigma_y0**2) / 2)
    """

    sigma0 = 5.0

    p0 = [amplitude0, xo0, yo0, sigma0]

    bounds = (
        [0, 0, 0, 1.0],
        [2000, nx, ny, 10.0],
    )

    evaluation_count = 0

    def counted_gaussian_2d(coords, amplitude, xo, yo, sigma):
        nonlocal evaluation_count
        evaluation_count += 1
        return gaussian_2d(coords, amplitude, xo, yo, sigma)

    start_time = time.perf_counter()
    popt, pcov = curve_fit(
        counted_gaussian_2d,
        (x, y),
        data.ravel(),
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    elapsed = time.perf_counter() - start_time

    print(f"curve_fit finished in {elapsed:.3f}s after {evaluation_count} model evaluations")

    return popt, pcov


def save_gaussian_fit_png(data, popt, out_path="gaussian_fit.png", waist_level="1e2"):
    """
    data: 2D numpy array
    popt: fit parameters from fit_gaussian_2d()
          [amplitude, xo, yo, sigma]
    out_path: output PNG filename
    waist_level:
        "sigma" -> circle radius = sigma
        "1e2"   -> circle radius = sqrt(2) * sigma
                  for the 1/e^2 radius of the Gaussian in the model used earlier
    """
    amplitude, xo, yo, sigma = popt

    if waist_level == "sigma":
        radius = sigma
    elif waist_level == "1e2":
        radius = np.sqrt(2) * sigma
    else:
        raise ValueError("waist_level must be 'sigma' or '1e2'")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Intensity")

    # Centre cross
    ax.plot(xo, yo, marker="+", markersize=18, markeredgewidth=0.5, color="red")

    # Waist circle
    circle = Circle((xo, yo), radius=radius, fill=False, color="red", linewidth=0.5)
    ax.add_patch(circle)

    ax.set_title("2D Gaussian Fit")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def gaussian_2d_image(shape, popt):
    amplitude, xo, yo, sigma = popt
    ny, nx = shape

    y = np.arange(ny)
    x = np.arange(nx)
    x, y = np.meshgrid(x, y)

    model = amplitude * np.exp(
        -(((x - xo)**2 + (y - yo)**2) / (2 * sigma**2))
    )
    return model


def save_gaussian_residual_png(data, popt, out_path="gaussian_residual.png"):
    data = np.asarray(data)
    model = gaussian_2d_image(data.shape, popt)
    residual = data - model

    fig, ax = plt.subplots(figsize=(6, 5))

    # Center the color scale on zero so positive/negative residuals stand out
    vmax = np.max(np.abs(residual))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(residual, origin="lower", cmap="seismic", norm=norm)
    fig.colorbar(im, ax=ax, label="Data - Model")

    ax.set_title("Residual Image")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def rotate_image(data):

    # interchange x-axis and y-axis
    data = np.rot90(data, k=-1)  # 90° clockwise

    return data


def analyse(image, freq_name, analysis_folder_path):
    # print image shape
    #print("Image shape:", image.shape)
    analyse_start = time.perf_counter()
    popt, pcov = fit_gaussian_2d(image)
    #print("amplitude, xo, yo, sigma =")
    #print(popt)
    #save_gaussian_fit_png(image, popt, f"{analysis_folder_path}/{freq_name}_fit.png")
    #save_gaussian_residual_png(image, popt, f"{analysis_folder_path}/{freq_name}_residual.png")
    print(f"analysis finished in {time.perf_counter() - analyse_start:.3f}s")

    return popt, pcov


images = read_pickle_file(f'{folder_path}\\raw.pkl')

# make analysis output folder
analysis_folder_path = f"{folder_path}\\analysis"
if not os.path.exists(analysis_folder_path):
    os.mkdir(analysis_folder_path)

# create a pickle file that will store the fit parameters for each frequency
fit_params_file_path = f"{folder_path}\\fit_params.pkl"
with open(fit_params_file_path, "wb") as f:
    pickle.dump({}, f)

for key in images.keys():
    print(key)
    image = images[key]
    #image = rotate_image(image)
    popt, pcov = analyse(image, key, analysis_folder_path)

    with open(fit_params_file_path, "rb") as f:
        fit_params = pickle.load(f)

    fit_params[key] = {
        "popt": popt,
        "pcov": pcov,
    }

    with open(fit_params_file_path, "wb") as f:
        pickle.dump(fit_params, f)
