import numpy as np


def _validate_grid(a, name):
    """
    Validate and convert a binary 2D grid.
    """
    a = np.asarray(a)

    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")

    if not np.isin(a, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1.")

    return a.astype(np.int8, copy=True)


def _apply_batch(state, moves):
    """
    Apply a simultaneous batch of moves.

    Each move is:

        ((src_row, src_col), (dst_row, dst_col))

    A destination may already contain an atom only if that atom is
    itself moved away in the same batch.
    """
    state = state.copy()

    if not moves:
        return state

    srcs = [tuple(src) for src, _ in moves]
    dsts = [tuple(dst) for _, dst in moves]

    if len(srcs) != len(set(srcs)):
        raise ValueError("Duplicate source cell in move batch.")

    if len(dsts) != len(set(dsts)):
        raise ValueError("Duplicate destination cell in move batch.")

    for src in srcs:
        if state[src] != 1:
            raise ValueError(f"Source {src} is empty.")

    src_set = set(srcs)

    for dst in dsts:
        if state[dst] == 1 and dst not in src_set:
            raise ValueError(
                f"Destination {dst} is occupied and is not "
                "vacated in this batch."
            )

    # Vacate all sources first.
    for src in srcs:
        state[src] = 0

    # Fill destinations.
    for dst in dsts:
        state[dst] = 1

    return state


def _choose_sources_for_destinations(
    source_positions,
    destination_positions,
):
    """
    Select only the required number of source atoms.

    Sources and destinations are 1D coordinates within a row or column.

    The selected sources satisfy:

    1. source order == destination order
       -> prevents trajectory crossing

    2. if a destination is already occupied, that atom must be selected
       -> prevents overwriting a stationary surplus atom

    3. among valid choices, minimize total displacement
    """
    sources = sorted(map(int, source_positions))
    dests = sorted(map(int, destination_positions))

    n = len(sources)
    m = len(dests)

    if m > n:
        raise ValueError(
            "Cannot fill more destinations than there are source atoms."
        )

    if m == 0:
        return []

    # Any atom already occupying a destination must participate.
    forced_sources = set(sources).intersection(dests)

    INF = 10**15

    # dp[i][j]:
    # minimum cost when considering first i sources
    # and assigning exactly j destinations.
    dp = [
        [INF] * (m + 1)
        for _ in range(n + 1)
    ]

    prev = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    dp[0][0] = 0

    for i in range(1, n + 1):

        source = sources[i - 1]

        for j in range(m + 1):

            # ----------------------------------------
            # Option 1: leave this atom unused
            #
            # Not allowed if it is sitting on one of
            # our intended destination cells.
            # ----------------------------------------

            if source not in forced_sources:

                if dp[i - 1][j] < dp[i][j]:

                    dp[i][j] = dp[i - 1][j]

                    prev[i][j] = (
                        i - 1,
                        j,
                        False,
                    )

            # ----------------------------------------
            # Option 2: use this source
            # ----------------------------------------

            if j > 0 and dp[i - 1][j - 1] < INF:

                destination = dests[j - 1]

                cost = (
                    dp[i - 1][j - 1]
                    + abs(source - destination)
                )

                if cost < dp[i][j]:

                    dp[i][j] = cost

                    prev[i][j] = (
                        i - 1,
                        j - 1,
                        True,
                    )

    if dp[n][m] >= INF:
        raise RuntimeError(
            "No collision-safe source subset could be selected."
        )

    # Reconstruct selected sources.
    chosen = []

    i = n
    j = m

    while i > 0 or j > 0:

        step = prev[i][j]

        if step is None:
            raise RuntimeError(
                "Failed to reconstruct source selection."
            )

        previous_i, previous_j, take = step

        if take:
            chosen.append(sources[i - 1])

        i = previous_i
        j = previous_j

    chosen.reverse()

    return chosen


def tetris_moves(initial, target, max_moves_per_batch=None):
    """
    Generate TETRIS rearrangement moves.

    Extra atoms in `initial` are allowed.

    Parameters
    ----------
    initial : 2D numpy array
        1 = occupied
        0 = empty

    target : 2D numpy array
        Target mask.

        Every target cell containing 1 must contain an atom at the end.

        Cells containing 0 may still contain surplus atoms.

    max_moves_per_batch : int or None
        Maximum number of moves allowed in each generated batch. If None,
        batches are not limited.

    Returns
    -------
    batches : list of dict
        Ordered simultaneous TETRIS move batches.

    final_state : np.ndarray
        State after applying all generated moves.

    extra_atoms : np.ndarray
        Coordinates of atoms that remain outside the target.

        Shape:
            (N_extra, 2)

        Each row is:
            [row, col]
    """

    if max_moves_per_batch is not None:
        if (
            not isinstance(max_moves_per_batch, (int, np.integer))
            or isinstance(max_moves_per_batch, (bool, np.bool_))
            or max_moves_per_batch < 1
        ):
            raise ValueError(
                "max_moves_per_batch must be a positive integer or None."
            )

    initial = _validate_grid(
        initial,
        "initial",
    )

    target = _validate_grid(
        target,
        "target",
    )

    # ------------------------------------------------------------
    # Target is assumed to be embedded in the same physical grid.
    # ------------------------------------------------------------

    if initial.shape != target.shape:
        raise ValueError(
            "initial and target must have the same shape. "
            "If the physical target is smaller, embed it into a "
            "full-size zero mask."
        )

    n_initial = int(initial.sum())
    n_target = int(target.sum())

    # ------------------------------------------------------------
    # Surplus atoms are fine.
    # Missing atoms are not.
    # ------------------------------------------------------------

    if n_initial < n_target:
        raise ValueError(
            f"Not enough atoms: initial has {n_initial}, "
            f"target needs {n_target}."
        )

    height, width = initial.shape

    state = initial.copy()

    batches = []

    # ============================================================
    # Target requirements
    #
    # pending[c] gives target rows in column c which still need
    # an atom assigned during the horizontal phase.
    # ============================================================

    pending = {
        c: [
            int(r)
            for r in np.flatnonzero(target[:, c])
        ]
        for c in range(width)
    }

    # ============================================================
    # PHASE 1
    #
    # TETRIMINO CONSTRUCTION
    #
    # Process rows from top to bottom.
    # ============================================================

    for r in range(height):

        source_cols = [
            int(c)
            for c in np.flatnonzero(state[r])
        ]

        incomplete_columns = [
            c
            for c in range(width)
            if pending[c]
        ]

        if not source_cols:
            continue

        if not incomplete_columns:
            # Target already has enough assigned atoms.
            # All remaining reservoir atoms are surplus.
            continue

        # --------------------------------------------------------
        # Priority:
        #
        # earliest missing target row first.
        #
        # Column number is deterministic tie-breaker.
        # --------------------------------------------------------

        incomplete_columns.sort(
            key=lambda c: (
                pending[c][0],
                c,
            )
        )

        # --------------------------------------------------------
        # IMPORTANT CHANGE:
        #
        # If there are more atoms in this row than needed,
        # only use the required subset.
        # --------------------------------------------------------

        n_use = min(
            len(source_cols),
            len(incomplete_columns),
        )

        chosen_target_columns = (
            incomplete_columns[:n_use]
        )

        # Priority chooses WHICH columns.
        #
        # Physical movement uses spatial ordering.
        destination_cols = sorted(
            chosen_target_columns
        )

        # --------------------------------------------------------
        # Choose the best subset of source atoms.
        #
        # Example:
        #
        # sources:
        #     0 1 2 5 7
        #
        # destinations:
        #     1 4 6
        #
        # Only three atoms are selected.
        # --------------------------------------------------------

        chosen_source_cols = (
            _choose_sources_for_destinations(
                source_cols,
                destination_cols,
            )
        )

        full_moves = [
            (
                (r, src_c),
                (r, dst_c),
            )
            for src_c, dst_c
            in zip(
                chosen_source_cols,
                destination_cols,
            )
        ]

        # Do not report zero-distance movements.
        actual_moves = [
            (src, dst)
            for src, dst in full_moves
            if src != dst
        ]

        if actual_moves:

            if (
                max_moves_per_batch is not None
                and len(actual_moves) > max_moves_per_batch
            ):
                raise RuntimeError(
                    "TETRIS batch exceeds max_moves_per_batch: "
                    f"{len(actual_moves)} > {max_moves_per_batch}."
                )

            state = _apply_batch(
                state,
                actual_moves,
            )

            batches.append({
                "phase": "horizontal",
                "row": r,
                "moves": actual_moves,
            })

        # --------------------------------------------------------
        # Every selected target column has received one assigned
        # atom.
        # --------------------------------------------------------

        for c in chosen_target_columns:
            pending[c].pop(0)

    # ============================================================
    # Check horizontal construction.
    # ============================================================

    remaining = {
        c: rows[:]
        for c, rows in pending.items()
        if rows
    }

    if remaining:

        raise RuntimeError(
            "TETRIS horizontal construction failed. "
            f"Unfilled target requirements: {remaining}"
        )

    # ============================================================
    # PHASE 2
    #
    # TETRIMINO ELIMINATION
    #
    # Each target column now has AT LEAST enough atoms.
    #
    # If a column has surplus atoms, select only enough atoms
    # to fill its target rows.
    # ============================================================

    for c in range(width):

        target_rows = [
            int(r)
            for r in np.flatnonzero(
                target[:, c]
            )
        ]

        if not target_rows:
            continue

        source_rows = [
            int(r)
            for r in np.flatnonzero(
                state[:, c]
            )
        ]

        if len(source_rows) < len(target_rows):

            raise RuntimeError(
                f"Column {c} has "
                f"{len(source_rows)} atoms but "
                f"needs {len(target_rows)}."
            )

        # --------------------------------------------------------
        # Select only the atoms required for this target column.
        #
        # Surplus atoms remain where they are.
        # --------------------------------------------------------

        chosen_source_rows = (
            _choose_sources_for_destinations(
                source_rows,
                target_rows,
            )
        )

        full_moves = [
            (
                (src_r, c),
                (dst_r, c),
            )
            for src_r, dst_r
            in zip(
                chosen_source_rows,
                target_rows,
            )
        ]

        actual_moves = [
            (src, dst)
            for src, dst in full_moves
            if src != dst
        ]

        if actual_moves:

            if (
                max_moves_per_batch is not None
                and len(actual_moves) > max_moves_per_batch
            ):
                raise RuntimeError(
                    "TETRIS batch exceeds max_moves_per_batch: "
                    f"{len(actual_moves)} > {max_moves_per_batch}."
                )

            state = _apply_batch(
                state,
                actual_moves,
            )

            batches.append({
                "phase": "vertical",
                "col": c,
                "moves": actual_moves,
            })

    # ============================================================
    # Verify that EVERY target site is occupied.
    # ============================================================

    missing = np.argwhere(
        (target == 1)
        & (state == 0)
    )

    if len(missing):

        raise RuntimeError(
            "Algorithm completed but target "
            f"is not filled: {missing.tolist()}"
        )

    # ------------------------------------------------------------
    # Anything occupied outside target is a surplus atom.
    # ------------------------------------------------------------

    extra_atoms = np.argwhere(
        (state == 1)
        & (target == 0)
    )

    return (
        batches,
        state,
        extra_atoms,
    )


def verify_tetris_moves(
    initial,
    target,
    batches,
    exact=False,
):
    """
    Replay and independently verify the move sequence.

    Parameters
    ----------
    exact : bool

        False:
            Success means every target site is occupied.
            Extra atoms are allowed.

        True:
            Final state must exactly equal target.

            This normally only makes sense when:
                initial.sum() == target.sum()

    Returns
    -------
    success
    final_state
    extra_atoms
    message
    """

    initial = _validate_grid(
        initial,
        "initial",
    )

    target = _validate_grid(
        target,
        "target",
    )

    if initial.shape != target.shape:

        return (
            False,
            initial,
            np.empty((0, 2), dtype=int),
            "Shape mismatch.",
        )

    state = initial.copy()

    height, width = state.shape

    # ============================================================
    # Replay each simultaneous batch.
    # ============================================================

    for batch_index, batch in enumerate(batches):

        phase = batch.get("phase")

        moves = batch.get(
            "moves",
            [],
        )

        # --------------------------------------------------------
        # Basic movement validation
        # --------------------------------------------------------

        for src, dst in moves:

            sr, sc = src
            dr, dc = dst

            if not (
                0 <= sr < height
                and 0 <= sc < width
                and 0 <= dr < height
                and 0 <= dc < width
            ):

                return (
                    False,
                    state,
                    np.empty((0, 2), dtype=int),
                    f"Batch {batch_index}: "
                    "coordinate out of bounds.",
                )

            if phase == "horizontal":

                if sr != dr:

                    return (
                        False,
                        state,
                        np.empty((0, 2), dtype=int),
                        f"Batch {batch_index}: "
                        "horizontal move changes row.",
                    )

            elif phase == "vertical":

                if sc != dc:

                    return (
                        False,
                        state,
                        np.empty((0, 2), dtype=int),
                        f"Batch {batch_index}: "
                        "vertical move changes column.",
                    )

            else:

                return (
                    False,
                    state,
                    np.empty((0, 2), dtype=int),
                    f"Batch {batch_index}: "
                    f"unknown phase {phase!r}.",
                )

        # --------------------------------------------------------
        # Crossing check
        # --------------------------------------------------------

        if phase == "horizontal":

            ordered = sorted(
                moves,
                key=lambda move: move[0][1],
            )

            destination_order = [
                dst[1]
                for _, dst in ordered
            ]

            if destination_order != sorted(
                destination_order
            ):

                return (
                    False,
                    state,
                    np.empty((0, 2), dtype=int),
                    f"Batch {batch_index}: "
                    "horizontal trajectories cross.",
                )

        elif phase == "vertical":

            ordered = sorted(
                moves,
                key=lambda move: move[0][0],
            )

            destination_order = [
                dst[0]
                for _, dst in ordered
            ]

            if destination_order != sorted(
                destination_order
            ):

                return (
                    False,
                    state,
                    np.empty((0, 2), dtype=int),
                    f"Batch {batch_index}: "
                    "vertical trajectories cross.",
                )

        # --------------------------------------------------------
        # Apply simultaneous operation.
        # --------------------------------------------------------

        try:

            state = _apply_batch(
                state,
                moves,
            )

        except Exception as exc:

            return (
                False,
                state,
                np.empty((0, 2), dtype=int),
                f"Batch {batch_index}: {exc}",
            )

    # ============================================================
    # Final verification
    # ============================================================

    extra_atoms = np.argwhere(
        (state == 1)
        & (target == 0)
    )

    if exact:

        success = np.array_equal(
            state,
            target,
        )

        if success:

            message = (
                "Verified: final state exactly "
                "matches target."
            )

        else:

            message = (
                "Final state does not exactly match "
                f"target; {len(extra_atoms)} "
                "surplus atoms remain."
            )

    else:

        missing = np.argwhere(
            (target == 1)
            & (state == 0)
        )

        success = len(missing) == 0

        if success:

            message = (
                "Verified: every target site is "
                f"filled; {len(extra_atoms)} "
                "surplus atoms remain."
            )

        else:

            message = (
                "Target is missing atoms at "
                f"{missing.tolist()}."
            )

    return (
        success,
        state,
        extra_atoms,
        message,
    )

