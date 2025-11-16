import numpy as np


def generate_signal(config, seed=7):
    """
    Generate a 1D signal composed of different segments, including
    constant, random, and PRBS segments.

    Parameters
    ----------
    config : dict
        Configuration dictionary defining all signal segments.

        Example
        -------
        >>> config = {
        ...     "segments": [
        ...         {"type": "constant", "value": 2.0, "points": 100},
        ...         {"type": "random", "value_range": [0, 5], "points": [50, 100]},
        ...         {"type": "prbs",
        ...          "points": 300,
        ...          "a_range": [0.0, 1.0],
        ...          "b_range": [5, 20]}
        ...     ],
        ...     "seed": 7,
        ... }

    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Generated 1D signal.
    """

    if seed is not None:
        np.random.seed(seed)

    signal = []

    for seg in config.get("segments", []):
        seg_type = seg.get("type", "").lower()

        # --- determine number of points ---
        points = seg["points"]
        if isinstance(points, (list, tuple)):
            n = np.random.randint(points[0], points[1] + 1)
        else:
            n = int(points)

        # --- handle each type of segment ---
        if seg_type == "constant":
            value = seg["value"]
            segment = np.full(n, value)

        elif seg_type == "random":
            vmin, vmax = seg["value_range"]
            value = np.random.uniform(vmin, vmax)
            segment = np.full(n, value)

        elif seg_type == "prbs":
            a_min, a_max = seg["a_range"]
            b_min, b_max = seg["b_range"]

            a = np.random.uniform(a_min, a_max, n)
            b = np.random.randint(b_min, b_max + 1, n)

            # Integrate b to determine step changes
            b[0] = 0
            for i in range(1, len(b)):
                b[i] += b[i - 1]

            prbs_signal = np.zeros(n)
            i = 0
            while b[i] < len(prbs_signal):
                k = b[i]
                prbs_signal[k:] = a[i]
                i += 1
                if i >= len(b):
                    break

            segment = prbs_signal

        else:
            raise ValueError(f"Unknown segment type '{seg_type}'")

        signal.extend(segment)

    return np.array(signal)
