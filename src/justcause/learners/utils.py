"""Miscellaneous tools used in the `justcause.learners`"""
from typing import Tuple

import numpy as np


def replace_factual_outcomes(
    y_0: np.array, y_1: np.array, y: np.array, t: np.array
) -> Tuple[np.array, np.array]:
    """ Replaces the predicted components with factual observations where possible

    Args:
        y_0: predicted control outcomes
        y_1: predicted treatment outcomes
        y: factual outcomes
        t: factual treatment indicators

    Returns: y_0, y_1 with factual outcomes replaced where possible
    """
    y_0 = np.where(t == 0, y, y_0)
    y_1 = np.where(t == 1, y, y_1)
    return y_0, y_1
