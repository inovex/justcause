from typing import List

import rpy2.robjects.packages as rpackages
from rpy2 import robjects
from rpy2.robjects import StrVector


def replace_factual_outcomes(y_0, y_1, y, t):
    """ Replaces the predicted components with factual observations where possible

    Args:
        y_0: predicted control outcomes
        y_1: predicted treatment outcomes
        y: factual outcomes
        t: factual treatment indicators

    Returns: y_0, y_1 with factual outcomes replaced where possible
    """
    for i in range(len(t)):
        if t[i] == 1:
            y_1[i] = y[i]
        else:
            y_0[i] = y[i]
    return y_0, y_1


def install_r_packages(package_names: List[str], verbose=False):
    """ Installs the packages if needed using rpy2 utility functions"""

    robjects.r.options(download_file_method="curl")
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install), verbose=verbose)
