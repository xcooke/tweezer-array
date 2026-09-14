import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from tetris_algorithm import tetris_moves, _apply_batch
from tetris_plot import plot_tetris_states
from trajectory import calculate_trajectory
from batches_to_states import convert_batches_to_states

def main():

    # ============================================================
    # Example reservoir
    #
    # More atoms than the target is allowed.
    # ============================================================

    array_size = 3
    
    # generate an array that is half filled
    initial = np.zeros((array_size, array_size), dtype=int)
    initial = np.random.choice(
        [0, 1],
        size=initial.shape,
        p=[0.5, 0.5],
    )

    print(f"Initial")
    print(initial)

    number_filled_sites = initial.sum()

    # ============================================================
    # Target mask
    #
    # Every 1 here must eventually contain an atom.
    # ============================================================

    # target array is size array_size x array_size of 0s
    # with a target_size x target_size square of 1s in the centre

    target_size = int(np.floor(np.sqrt(number_filled_sites)))


    target = np.zeros((array_size, array_size), dtype=int)
    start_row = (array_size - target_size) // 2
    start_col = (array_size - target_size) // 2
    target[start_row:start_row + target_size, start_col:start_col + target_size] = 1

    print("Target:")
    print(target)

    # ============================================================
    # Generate TETRIS moves.
    # ============================================================

    batches, final_state, extra_atoms = tetris_moves(
        initial,
        target,
        5,
    )

    print(f"Initial atoms: {initial.sum()}")
    print(f"Target atoms:  {target.sum()}")
    print(f"Extra atoms:   {len(extra_atoms)}")
    print(f"Move batches:  {len(batches)}")

    print("\nExtra atoms:")
    print(extra_atoms)

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

    sites = {}
    for row in range(array_size):
        for col in range(array_size):
            sites[(col, row)] = (row, col)

    states = convert_batches_to_states(
        batches,
        sites,
        1.0,
        0.2,
        "characterisations\\2026-09-04_14-21-52\\position_calibration.pkl",
    )

    print(states.shape)
    print(states.shape[0])


if __name__ == "__main__":
    main()