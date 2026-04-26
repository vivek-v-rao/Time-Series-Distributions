""" shared utilities for comparing fitted distributions """
import numpy as np


def count_fit_params(dist_name, dist_fit):
    """Return the effective parameter count for a named distribution fit."""
    if dist_name == "normal":
        return 1
    if dist_name == "GED":
        return 2
    if dist_name in ("SGED", "NIG", "skew-t"):
        return 3
    if dist_name == "GH":
        return 4
    if dist_name == "t":
        return 2
    if dist_name.startswith("mix") or dist_name.startswith("hypsec"):
        return 3 * dist_fit["n_components"] - 1
    return np.nan
