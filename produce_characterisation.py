import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


# ============================================================
# Configuration
# ============================================================


folder = Path(r'characterisations\2026-09-09_17-39-02')

file_path = folder / "fit_params.pkl"
output_path = folder / "position_calibration.pkl"


# ============================================================
# Load fitted calibration data
# ============================================================

with open(file_path, "rb") as f:
    data = pickle.load(f)


# ============================================================
# Extract calibration points
# ============================================================

def extract_calibration_data(data):
    """
    Extract arrays from fit_params.pkl.

    Assumes:
        key = (x_freq, y_freq)
        popt = (amplitude, x_position, y_position, sigma)

    Returns
    -------
    freqs : (N, 2)
        [x_freq, y_freq]

    positions : (N, 2)
        [x_position, y_position]

    amplitudes : (N,)
        measured fitted amplitudes
    """

    freqs = []
    positions = []
    amplitudes = []

    for (x_freq, y_freq), result in data.items():

        amplitude, x_pos, y_pos, sigma = result["popt"]

        freqs.append([x_freq, y_freq])
        positions.append([x_pos, y_pos])
        amplitudes.append(amplitude)

    return (
        np.asarray(freqs, dtype=float),
        np.asarray(positions, dtype=float),
        np.asarray(amplitudes, dtype=float),
    )


freqs, positions, amplitudes = extract_calibration_data(data)


# ============================================================
# 1. VISUALISATIONS
# ============================================================

def plot_amplitude_heatmap(freqs, amplitudes):
    """
    Plot fitted amplitude as a function of x/y frequency.
    """

    x_freqs = np.unique(freqs[:, 0])
    y_freqs = np.unique(freqs[:, 1])

    heatmap = np.full(
        (len(y_freqs), len(x_freqs)),
        np.nan
    )

    for (x_freq, y_freq), amplitude in zip(freqs, amplitudes):

        ix = np.where(x_freqs == x_freq)[0][0]
        iy = np.where(y_freqs == y_freq)[0][0]

        heatmap[iy, ix] = amplitude

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        heatmap,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=[
            x_freqs.min(),
            x_freqs.max(),
            y_freqs.min(),
            y_freqs.max(),
        ],
    )

    ax.set_xlabel("x frequency (MHz)")
    ax.set_ylabel("y frequency (MHz)")
    ax.set_title("Measured amplitude")

    fig.colorbar(
        im,
        ax=ax,
        label="Amplitude"
    )

    plt.tight_layout()

    # save figure

    plt.savefig(f"{folder}/analysis/amplitude_heatmap.png", dpi=300, bbox_inches="tight")


def plot_warped_grid(freqs, positions):
    """
    Plot lines of constant x-frequency and constant y-frequency
    in physical position space.

    This directly shows the frequency -> position warping.
    """

    fig, ax = plt.subplots(figsize=(7, 7))

    x_freqs = np.unique(freqs[:, 0])
    y_freqs = np.unique(freqs[:, 1])

    # Lines of constant x frequency
    for x_freq in x_freqs:

        mask = np.isclose(freqs[:, 0], x_freq)

        pts = positions[mask]
        corresponding_y_freqs = freqs[mask, 1]

        order = np.argsort(corresponding_y_freqs)

        ax.plot(
            pts[order, 0],
            pts[order, 1],
            "k-",
            linewidth=0.8
        )

    # Lines of constant y frequency
    for y_freq in y_freqs:

        mask = np.isclose(freqs[:, 1], y_freq)

        pts = positions[mask]
        corresponding_x_freqs = freqs[mask, 0]

        order = np.argsort(corresponding_x_freqs)

        ax.plot(
            pts[order, 0],
            pts[order, 1],
            "k-",
            linewidth=0.8
        )

    # Show measured points as well
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=10
    )

    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_title("Warped frequency grid")

    ax.axis("equal")

    plt.tight_layout()

    plt.savefig(f"{folder}/analysis/warped_grid.png", dpi=300, bbox_inches="tight")


# ============================================================
# 2. BUILD POSITION -> FREQUENCY / AMPLITUDE CALIBRATION
# ============================================================

def build_calibration(freqs, positions, amplitudes):
    """
    Build interpolation maps:

        desired position (x, y)
                |
                +--> x frequency
                +--> y frequency
                +--> measured response

    The amplitude correction is normalised so that the weakest
    location uses drive amplitude = 1.0.

    Therefore all requested amplitudes are <= 1.
    """

    x_freq_interp = LinearNDInterpolator(
        positions,
        freqs[:, 0],
        fill_value=np.nan
    )

    y_freq_interp = LinearNDInterpolator(
        positions,
        freqs[:, 1],
        fill_value=np.nan
    )

    amplitude_interp = LinearNDInterpolator(
        positions,
        amplitudes,
        fill_value=np.nan
    )

    # Weakest measured response determines the maximum
    # uniform output achievable over the whole plane.
    reference_amplitude = np.nanmin(amplitudes)

    return {
        "x_freq_interp": x_freq_interp,
        "y_freq_interp": y_freq_interp,
        "amplitude_interp": amplitude_interp,

        "freqs": freqs,
        "positions": positions,
        "amplitudes": amplitudes,

        "reference_amplitude": reference_amplitude,
    }

calibration = build_calibration(
    freqs,
    positions,
    amplitudes
)


# ============================================================
# Query function
# ============================================================

def query_calibration(calibration, position):

    position = np.asarray(position, dtype=float)

    if position.shape != (2,):
        raise ValueError("position must be [x, y]")

    p = position[None, :]

    x_freq = float(
        calibration["x_freq_interp"](p)[0]
    )

    y_freq = float(
        calibration["y_freq_interp"](p)[0]
    )

    measured_amplitude = float(
        calibration["amplitude_interp"](p)[0]
    )

    if not np.all(
        np.isfinite([x_freq, y_freq, measured_amplitude])
    ):
        raise ValueError(
            f"Position {position} is outside the calibrated region."
        )

    if measured_amplitude <= 0:
        raise ValueError(
            "Interpolated amplitude is zero or negative."
        )

    reference_amplitude = calibration["reference_amplitude"]

    # Normalisation:
    #
    # weakest point -> 1.0
    # stronger points -> < 1.0
    amplitude = reference_amplitude / measured_amplitude

    # Protect against numerical interpolation errors
    amplitude = np.clip(amplitude, 0.0, 1.0)

    return x_freq, y_freq, amplitude

# ============================================================
# Save calibration
# ============================================================

with open(output_path, "wb") as f:
    pickle.dump(calibration, f)

print(f"Saved calibration to:")
print(output_path)


# ============================================================
# Visualise calibration
# ============================================================

plot_amplitude_heatmap(
    freqs,
    amplitudes
)

plot_warped_grid(
    freqs,
    positions
)

plt.show()


# ============================================================
# Example
# ============================================================

x_freq, y_freq, amplitude = query_calibration(
    calibration,
    [480, 755]
)

print(f"x frequency : {x_freq:.3f} MHz")
print(f"y frequency : {y_freq:.3f} MHz")
print(f"amplitude   : {amplitude:.3f}")