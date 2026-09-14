import time
from math import ceil, sqrt

import matplotlib.pyplot as plt
import numpy as np

from tetris_algorithm import tetris_moves


# ============================================================
# Benchmark configuration
# ============================================================

# Target atom numbers N must be perfect squares because the
# requested target is sqrt(N) x sqrt(N).

# N_VALUES be list of perfect squares from 4 to 10000

N_VALUES = [x**2 for x in range(2, 101, 4)]

# Number of random initial configurations used for each N.
N_TRIALS = 10

# Initial filling fraction. 0.50 means exactly half the reservoir
# sites are populated (up to the reservoir-size construction below).
FILLING_FRACTION = 0.50

# Reproducible random-number generator.
RNG_SEED = 12345

# Output figure names.
MOVE_PLOT = "tetris_moves_vs_N.png"
TIME_PLOT = "tetris_time_vs_N.png"


# ============================================================
# Array construction
# ============================================================


def make_target(N):
    """
    Make a sqrt(N) x sqrt(N) square target centred in a square
    reservoir array.

    The reservoir side length is chosen so that there are at least
    twice as many sites as target atoms. This allows an exactly 50%
    filled reservoir to contain at least N atoms.
    """
    L = int(sqrt(N))

    if L * L != N:
        raise ValueError(
            f"N={N} is not a perfect square. "
            """Use values such as 4, 9, 16, 25, ... ."""
        )

    # Smallest square reservoir with >= 2N sites.
    reservoir_side = ceil(sqrt(2 * N))

    # Make the side even so that an exact 50% filling is possible.
    if (reservoir_side * reservoir_side) % 2 != 0:
        reservoir_side += 1

    target = np.zeros(
        (reservoir_side, reservoir_side),
        dtype=np.int8,
    )

    start = (reservoir_side - L) // 2
    stop = start + L

    target[start:stop, start:stop] = 1

    return target


def make_initial(target, rng, filling=0.50):
    """
    Generate a random initial array with an exact filling fraction.

    The array has the same shape as target. Exactly filling * M sites
    are populated, where M is the total number of reservoir sites.
    """
    shape = target.shape
    n_sites = int(np.prod(shape))

    n_atoms = int(round(filling * n_sites))

    if n_atoms < int(target.sum()):
        raise ValueError(
            "The requested filling does not provide enough atoms "
            "for the target."
        )

    flat = np.zeros(n_sites, dtype=np.int8)
    occupied = rng.choice(
        n_sites,
        size=n_atoms,
        replace=False,
    )
    flat[occupied] = 1

    return flat.reshape(shape)


# ============================================================
# Move counting
# ============================================================


def count_move_batches(batches):
    """
    Count parallel operation batches.
    """
    return len(batches)


# ============================================================
# Benchmark
# ============================================================


def benchmark_tetris(
    N_values,
    n_trials,
    filling,
    seed,
):
    """
    Benchmark TETRIS over random initial configurations.

    Returns a dictionary containing:

        N
        target_side
        reservoir_side
        successful_trials
        failed_trials
        median_moves
        max_moves
        median_batches
        max_batches
        median_time_s
        max_time_s

    Timing measures only the call to tetris_moves().
    """
    rng = np.random.default_rng(seed)

    results = []

    for N in N_values:

        target = make_target(N)
        reservoir_side = target.shape[0]

        move_counts = []
        batch_counts = []
        calculation_times = []

        failures = 0

        print(
            f"N={N:4d} | target={int(sqrt(N)):2d}x{int(sqrt(N)):2d} "
            f"| reservoir={reservoir_side}x{reservoir_side}",
            end="",
            flush=True,
        )

        for trial in range(n_trials):

            initial = make_initial(
                target,
                rng,
                filling=filling,
            )

            # Time ONLY the calculation of the move list.
            start_time = time.perf_counter()

            try:
                batches, final_state, extra_atoms = tetris_moves(
                    initial,
                    target,
                )

                elapsed = time.perf_counter() - start_time

            except RuntimeError:
                # TETRIS can fail for a particular stochastic loading
                # even when there are enough atoms overall.
                failures += 1
                continue

            # Sanity check: all target sites must be occupied.
            target_filled = np.all(
                final_state[target == 1] == 1
            )

            if not target_filled:
                failures += 1
                continue

            # One simultaneous horizontal or vertical batch counts
            # as ONE TETRIS move, regardless of how many atoms move
            # within that batch.
            batch_counts.append(
                count_move_batches(batches)
            )

            calculation_times.append(
                elapsed
            )

        if not batch_counts:
            raise RuntimeError(
                f"No successful TETRIS trials for N={N}."
            )

        result = {
            "N": N,
            "target_side": int(sqrt(N)),
            "reservoir_side": reservoir_side,
            "successful_trials": len(batch_counts),
            "failed_trials": failures,
            "min_moves": int(np.min(batch_counts)),
            "median_moves": float(np.median(batch_counts)),
            "max_moves": int(np.max(batch_counts)),
            "min_time_s": float(np.min(calculation_times)),
            "median_time_s": float(np.median(calculation_times)),
            "max_time_s": float(np.max(calculation_times)),
        }

        results.append(result)

        print(
            f" | success={len(batch_counts):3d}/{n_trials}"
            f" | median batches={result['median_moves']:7.1f}"
            f" | max batches={result['max_moves']:7d}"
            f" | median time={result['median_time_s']:.6g} s"
        )

    return results


# ============================================================
# Plotting
# ============================================================


def plot_moves(results, filename):
    """
    Plot median number of TETRIS batches with minimum-to-maximum bars.

    Each batch is one parallel TETRIS operation, regardless of the
    number of atoms moved within that batch.
    """
    N = np.array([r["N"] for r in results])
    minimum = np.array([r["min_moves"] for r in results])
    median = np.array([r["median_moves"] for r in results])
    maximum = np.array([r["max_moves"] for r in results])

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        N,
        median,
        yerr=np.vstack((median - minimum, maximum - median)),
        fmt="o",
        capsize=4,
        color="C0",
        label="Median with minimum-to-maximum range",
    )

    ax.set_xlabel("Number of target atoms, N")
    ax.set_ylabel("Number of TETRIS move batches")
    ax.set_title("TETRIS rearrangement batches")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_time(results, filename):
    """
    Plot median calculation time with minimum-to-maximum bars.
    """
    N = np.array([r["N"] for r in results])
    minimum_time = np.array([
        r["min_time_s"]
        for r in results
    ])
    median_time = np.array([
        r["median_time_s"]
        for r in results
    ])
    maximum_time = np.array([
        r["max_time_s"]
        for r in results
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        N,
        median_time,
        yerr=np.vstack((median_time - minimum_time, maximum_time - median_time)),
        fmt="o",
        capsize=4,
        color="C0",
        label="Median with minimum-to-maximum range",
    )

    ax.set_xlabel("Number of target atoms, N")
    ax.set_ylabel("TETRIS calculation time (s)")
    ax.set_title("TETRIS calculation time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main():

    results = benchmark_tetris(
        N_values=N_VALUES,
        n_trials=N_TRIALS,
        filling=FILLING_FRACTION,
        seed=RNG_SEED,
    )

    plot_moves(
        results,
        MOVE_PLOT,
    )

    plot_time(
        results,
        TIME_PLOT,
    )

    print("\nResults")
    print("-------")

    for r in results:
        print(
            f"N={r['N']:4d} | "
            f"median batches={r['median_moves']:8.1f} | "
            f"max batches={r['max_moves']:8d} | "
            f"median time={r['median_time_s']:.6g} s | "
            f"success={r['successful_trials']}/{N_TRIALS}"
        )

    print(f"\nSaved: {MOVE_PLOT}")
    print(f"Saved: {TIME_PLOT}")


if __name__ == "__main__":
    main()
