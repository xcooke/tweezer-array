import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def draw_state(ax, state, target, title="", moves=None):
    """
    Draw one array state.

    Parameters
    ----------
    ax : matplotlib Axes
    state : 2D numpy array
        Current atom occupancy.
    target : 2D numpy array
        Target occupancy mask.
    title : str
        Plot title.
    moves : list
        List of:
            ((src_row, src_col), (dst_row, dst_col))
    """

    rows, cols = state.shape

    # 0 = empty
    # 1 = occupied
    cmap = ListedColormap([
        "white",
        "black",
    ])

    ax.imshow(
        state,
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="upper",
    )

    # ------------------------------------------------------------
    # Draw target sites as red boxes.
    # ------------------------------------------------------------
    for r, c in np.argwhere(target == 1):
        rect = plt.Rectangle(
            (c - 0.45, r - 0.45),
            0.9,
            0.9,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )

        ax.add_patch(rect)

    # ------------------------------------------------------------
    # Draw arrows for moves in this batch.
    # ------------------------------------------------------------
    if moves:
        for src, dst in moves:

            sr, sc = src
            dr, dc = dst

            ax.annotate(
                "",
                xy=(dc, dr),
                xytext=(sc, sr),
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=2,
                    color="blue",
                ),
            )

    # Grid.
    ax.set_xticks(
        np.arange(-0.5, cols, 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, rows, 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="gray",
        linewidth=0.7,
    )

    # Coordinate labels.
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))

    ax.set_xlabel("column")
    ax.set_ylabel("row")

    ax.set_title(title)

    # Prevent minor ticks from sticking out.
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )


def plot_tetris_states(states, target, batches, output_file="tetris_moves.png"):
    """Plot the initial state and every state after a move batch."""

    n_panels = len(states)
    ncols = min(4, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            4 * ncols,
            4 * nrows,
        ),
        squeeze=False,
    )

    axes = axes.flatten()

    draw_state(
        axes[0],
        states[0],
        target,
        title="Initial state",
    )

    for i, batch in enumerate(
        batches,
        start=1,
    ):

        phase = batch["phase"]

        if phase == "horizontal":

            label = (
                f"Move {i}\n"
                f"horizontal row {batch['row']}"
            )

        else:

            label = (
                f"Move {i}\n"
                f"vertical column {batch['col']}"
            )

        draw_state(
            axes[i],
            states[i],
            target,
            title=label,
            moves=batch["moves"],
        )

    for i in range(
        n_panels,
        len(axes),
    ):
        axes[i].axis("off")

    fig.suptitle(
        "TETRIS rearrangement",
        fontsize=18,
    )

    fig.tight_layout()
    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"\nSaved visualization to {output_file}"
    )