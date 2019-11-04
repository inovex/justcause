import numpy as np
import rpy2.robjects.packages as rpackages

from justcause.data.utils import iter_rep
from justcause.learners.utils import install_r_packages, replace_factual_outcomes


def test_iter_rep(dummy_df):
    assert "rep" in dummy_df.columns
    single_rep = next(iter_rep(dummy_df))
    assert "rep" not in single_rep.columns
    assert single_rep.shape[0] == 5


def test_replace_factuals():
    y_0 = np.zeros(10)
    y_1 = np.ones(10)
    y = np.repeat(0.5, 10)
    t = np.zeros(10)
    t[5] = 1
    y_0, y_1 = replace_factual_outcomes(y_0, y_1, y, t)
    assert y_1[5] == y[5]
    assert y_1[0] != y[0]
    assert y_0[0] == y[0]


def test_install_r_packages(uninstall_grf):
    package_names = ["grf"]
    install_r_packages(package_names, verbose=True)
    assert rpackages.isinstalled(package_names[0])
