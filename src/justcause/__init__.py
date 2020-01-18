from pkg_resources import get_distribution, DistributionNotFound

from . import learners
from . import data
from .data import CausalFrame

__all__ = ["learners", "data", "CausalFrame"]

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
