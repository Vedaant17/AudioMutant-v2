import numpy as np


def extract_mid_side(y):

    if y.ndim < 2:
        return {
            "mid_energy": 0.0,
            "side_energy": 0.0,
            "side_ratio": 0.0,
            "mid_side_balance": 0.0
        }

    left = y[0]
    right = y[1]

    # -----------------------------
    # MID / SIDE
    # -----------------------------
    mid = (left + right) / 2
    side = (left - right) / 2

    mid_energy = float(np.mean(np.abs(mid)))
    side_energy = float(np.mean(np.abs(side)))

    # -----------------------------
    # SPATIAL METRICS
    # -----------------------------
    total = mid_energy + side_energy + 1e-6

    side_ratio = float(side_energy / total)
    mid_side_balance = float(mid_energy - side_energy)

    return {
        "mid_energy": mid_energy,
        "side_energy": side_energy,
        "side_ratio": side_ratio,
        "mid_side_balance": mid_side_balance
    }