import numpy as np
import pickle

from drivers.trajectory import calculate_trajectory
from drivers.aom_use_calibration import find_aom_spot_parameters

def convert_batches_to_states(batches, sites, T, dt, aom_calibration_file):

    # now convert into coordinates
    
    # states has shape:
    # (num_steps, 10, 2)
    #
    # states[i, oscillator, 0] = frequency in MHz
    # states[i, oscillator, 1] = amplitude
    #
    # oscillators 0-4 -> Phaser channel 0 (x)
    # oscillators 5-9 -> Phaser channel 1 (y)

    with open(aom_calibration_file, "rb") as f:
        aom_calibration = pickle.load(f)

    states = []

    for batch in batches:
        num_steps = int(T / dt)
        
        moves = batch["moves"]

        batch_states = np.zeros((num_steps+1, 10, 2), dtype=np.float64)


        if batch["phase"] == "horizontal":
            row = batch["row"]
            row_x, row_y = sites[(row, 0)] # this depends on which column... will not be perfect

            _, row_y_freq, _ = find_aom_spot_parameters(aom_calibration, row_x, row_y, 1.0)

            batch_states[:, 5, 0] = row_y_freq
            batch_states[:, 5, 1] = 1.0

            # i from 0 to 4
            for i, (src, dst) in enumerate(moves):
                d = sites[dst][0] - sites[src][0]
                displacement_steps, amplitude_steps = calculate_trajectory(d, T, dt, num_steps)
                x_steps = [sites[src][0] + step for step in displacement_steps]

                for j in range(len(x_steps)):
                    x_step = x_steps[j]
                    amplitude_step = amplitude_steps[j]
                    x_freq, y_freq, amplitude = find_aom_spot_parameters(aom_calibration, x_step, row_y, amplitude_step)
                    batch_states[j, i, 0] = x_freq
                    batch_states[j, i, 1] = amplitude
                
        elif batch["phase"] == "vertical":
            col = batch["col"]
            col_x, col_y = sites[(0, col)] # this depends on which row... will not be perfect

            col_x_freq, _, _ = find_aom_spot_parameters(aom_calibration, col_x, col_y, 1.0)

            batch_states[:, 0, 0] = col_x_freq
            batch_states[:, 0, 1] = 1.0

            # i from 5 to 9
            for i, (src, dst) in enumerate(moves):
                d = sites[dst][1] - sites[src][1]
                displacement_steps, amplitude_steps = calculate_trajectory(d, T, dt, num_steps)
                y_steps = [sites[src][1] + step for step in displacement_steps]

                for j in range(len(y_steps)):
                    y_step = y_steps[j]
                    amplitude_step = amplitude_steps[j]
                    x_freq, y_freq, amplitude = find_aom_spot_parameters(aom_calibration, col_x, y_step, amplitude_step)
                    batch_states[j, i + 5, 0] = y_freq
                    batch_states[j, i + 5, 1] = amplitude

        states.append(batch_states if not states else batch_states)

    final_states = np.concatenate(states, axis=0) if states else np.empty((0, 10, 2))

    return final_states