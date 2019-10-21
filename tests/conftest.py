# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for justcause.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import pytest

from justcause.data.sets.ibm import load_ibm_acic
from justcause.data.sets.ihdp import load_ihdp
from justcause.data.sets.twins import load_twins


@pytest.fixture
def grf():
    """Assure the installation of Generalized Random Forests"""
    from justcause.methods.causal_forest import CausalForest

    CausalForest.install_grf()


@pytest.fixture
def ihdp_data():
    return load_ihdp()


@pytest.fixture
def ibm_data():
    return load_ibm_acic()


@pytest.fixture
def twins_data():
    return load_twins()
