"""Bundles all synthetic data generating processes (DGP)

Exported are four data generating processes, which are either used in the thesis or
used in reputable papers. More DGPs are always desirable, so feel free to contribute
new implementation to this module.

"""
from .ihdp import multi_expo_on_ihdp
from .toy import toy_data_synthetic, toy_data_emcs
from .rlearner import rlearner_simulation_data

__all__ = [
    "multi_expo_on_ihdp",
    "toy_data_synthetic",
    "toy_data_emcs",
    "rlearner_simulation_data",
]
