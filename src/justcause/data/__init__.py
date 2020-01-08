"""Contains reference data sets and helpers for parametric generation of new data.

Under .sets, common reference data sets like IHDP or Twins are provided in an easily
accessible and reproducible format.
In the .generators module, implementations of data generation processes used in the
thesis and in reputable papers can be accesssed.
The Frames module introduces the CausalFrame, a specialization of the pandas.DataFrame
and the utils contain tools to generate your own data based on a clear and
comprehensive convention.

"""
from . import sets
from . import generators

from .frames import CausalFrame, Col

__all__ = ["sets", "generators", "CausalFrame", "Col"]
