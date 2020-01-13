"""Miscellaneous tools used in the `justcause.learners`"""
from typing import List, Tuple

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


def install_r_packages(package_names: List[str], verbose=False):
    """Installs the R packages if needed using rpy2 utility functions

    Args:
        package_names: names of the R packages to install
        verbose: Whether to print progress information or not

    """
    import rpy2.robjects.packages as rpackages
    from rpy2 import robjects
    from rpy2.robjects import StrVector

    robjects.r.options(download_file_method="curl")
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install), verbose=verbose)
