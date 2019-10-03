# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for justcause.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import pytest


@pytest.fixture
def grf():
    """Assure the installation of Generalized Random Forests"""
    from justcause.methods.causal_forest import CausalForest
    CausalForest.install_grf()
