import pickle
import numpy as np


with open("characterisations\\2026-08-27_15-47-18\\position_calibration.pkl", "rb") as f:
    calibration = pickle.load(f)


def get_drive_parameters(calibration, x, y):

    p = np.array([[x, y]], dtype=float)

    x_freq = float(calibration["x_freq_interp"](p)[0])
    y_freq = float(calibration["y_freq_interp"](p)[0])

    local_amp = float(
        calibration["amplitude_interp"](p)[0]
    )

    amplitude = (
        calibration["reference_amplitude"]
        / local_amp
    )

    return x_freq, y_freq, amplitude


x_freq, y_freq, amplitude = get_drive_parameters(
    calibration,
    x=110,
    y=86
)

print("x frequency:", x_freq)
print("y frequency:", y_freq)
print("amplitude:", amplitude)