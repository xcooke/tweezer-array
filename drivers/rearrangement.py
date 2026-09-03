import numpy as np
import heapq
from functools import lru_cache
from scipy.optimize import linear_sum_assignment


def make_central_target(shape, n_atoms):
    """
    Pick n_atoms sites closest to the geometric centre of the array.
    Returns a boolean array representing the desired final occupancy.
    """
    rows, cols = shape
    cr = (rows - 1) / 2
    cc = (cols - 1) / 2

    sites = [(r, c) for r in range(rows) for c in range(cols)]

    # Distance from centre.
    # Tie breakers make the result deterministic.
    sites.sort(key=lambda p: (
        (p[0] - cr)**2 + (p[1] - cc)**2,
        abs(p[0] - cr) + abs(p[1] - cc),
        p[0],
        p[1]
    ))

    target = np.zeros(shape, dtype=bool)

    for r, c in sites[:n_atoms]:
        target[r, c] = True

    return target


def configuration_to_tuple(array):
    """Convert occupancy array to hashable tuple of occupied coordinates."""
    return tuple(map(tuple, np.argwhere(array)))


def tuple_to_set(state):
    return set(state)


def solve_rearrangement(initial):
    """
    Rearrange atoms into a compact central blob using only
    nearest-neighbour horizontal/vertical moves.

    Parameters
    ----------
    initial : 2D array-like
        1 = occupied
        0 = empty

    Returns
    -------
    moves : list
        Ordered list:
            [((r1,c1), (r2,c2)), ...]
        Every move is guaranteed to be into an empty neighbouring site.

    target : numpy.ndarray
        Desired final occupancy.
    """

    initial = np.asarray(initial, dtype=bool)
    rows, cols = initial.shape
    n_atoms = int(initial.sum())

    target = make_central_target(initial.shape, n_atoms)

    start = configuration_to_tuple(initial)
    goal = configuration_to_tuple(target)

    target_positions = np.array(goal, dtype=int)

    # ---------------------------------------------------------
    # Heuristic:
    # Minimum total Manhattan distance from atoms to targets.
    #
    # Hungarian algorithm finds the best atom-target assignment.
    # ---------------------------------------------------------

    @lru_cache(maxsize=None)
    def heuristic(state):
        if state == goal:
            return 0

        atoms = np.array(state, dtype=int)

        cost = (
            np.abs(atoms[:, None, 0] - target_positions[None, :, 0])
            +
            np.abs(atoms[:, None, 1] - target_positions[None, :, 1])
        )

        row_ind, col_ind = linear_sum_assignment(cost)

        return int(cost[row_ind, col_ind].sum())

    # ---------------------------------------------------------
    # Generate all legal single-site moves
    # ---------------------------------------------------------

    directions = [
        (-1, 0),
        (+1, 0),
        (0, -1),
        (0, +1),
    ]

    def neighbours(state):
        occupied = set(state)

        for r, c in state:
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc

                # Must remain on grid
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                # Destination must be empty
                if (nr, nc) in occupied:
                    continue

                new_occupied = occupied.copy()
                new_occupied.remove((r, c))
                new_occupied.add((nr, nc))

                new_state = tuple(sorted(new_occupied))

                move = ((r, c), (nr, nc))

                yield new_state, move

    # ---------------------------------------------------------
    # A* search
    # ---------------------------------------------------------

    # Entries:
    # (f = g+h, g, unique_counter, state)

    counter = 0
    pq = [(heuristic(start), 0, counter, start)]

    best_g = {start: 0}

    # child state -> (parent state, move)
    came_from = {}

    while pq:

        f, g, _, state = heapq.heappop(pq)

        # Ignore stale queue entries
        if g != best_g.get(state):
            continue

        # Found solution
        if state == goal:
            break

        for new_state, move in neighbours(state):

            new_g = g + 1

            if new_g < best_g.get(new_state, float("inf")):

                best_g[new_state] = new_g
                came_from[new_state] = (state, move)

                counter += 1

                new_f = new_g + heuristic(new_state)

                heapq.heappush(
                    pq,
                    (new_f, new_g, counter, new_state)
                )

    else:
        raise RuntimeError("No solution found.")

    # ---------------------------------------------------------
    # Reconstruct ordered sequence of moves
    # ---------------------------------------------------------

    moves = []

    state = goal

    while state != start:
        previous_state, move = came_from[state]
        moves.append(move)
        state = previous_state

    moves.reverse()

    return moves, target


def apply_moves(initial, moves):
    """
    Apply a list of moves, checking that every move is legal.
    Useful for verifying the result.
    """

    state = np.asarray(initial, dtype=int).copy()

    rows, cols = state.shape

    for i, (source, destination) in enumerate(moves):

        r1, c1 = source
        r2, c2 = destination

        # Check nearest neighbour
        assert abs(r1 - r2) + abs(c1 - c2) == 1, \
            f"Move {i}: not nearest-neighbour"

        # Check source occupied
        assert state[r1, c1] == 1, \
            f"Move {i}: source {source} is empty"

        # Check destination empty
        assert state[r2, c2] == 0, \
            f"Move {i}: destination {destination} is occupied"

        state[r1, c1] = 0
        state[r2, c2] = 1

    return state

