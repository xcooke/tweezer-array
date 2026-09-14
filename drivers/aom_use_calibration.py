import numpy as np

def find_aom_spot_parameters(calibration, x, y, amp):
    #print("find_aom_spot_parameters called with x:", x, "y:", y, "amp:", amp)
    p = np.array([[x, y]], dtype=float)

    x_freq = float(calibration["x_freq_interp"](p)[0])
    y_freq = float(calibration["y_freq_interp"](p)[0])

    local_amp = float(
        calibration["amplitude_interp"](p)[0]
    )

    amplitude_modification = (
        calibration["reference_amplitude"]
        / local_amp
    )

    amplitude_out = np.sqrt(amp * amplitude_modification)

    if x_freq is np.nan:
        return 0.0, 0.0, 1.0
    #print(x_freq-100, y_freq-100, amplitude_out)

    return x_freq-100, y_freq-100, amplitude_out
    #return x, y, amp