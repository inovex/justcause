"""
Generalised DataFrame which differentiates between covariates and other column names
"""
# Uncomment only when we require Python >= 3.7
# from __future__ import annotations

from typing import List, Type

import numpy as np
import pandas as pd
from pandas.core.internals import BlockManager


class Col:
    """Column names of a CausalFrame

    Example:
        To access the untreated noiseless outcome column mu_0 in a CausalFrame cf,
        write cf[Col.mu_0] to avoid misspelling and to robust against changes in the
        terminology, which will we added here.
    """

    t = "t"
    y = "y"
    y_cf = "y_cf"
    y_0 = "y_0"
    y_1 = "y_1"
    mu_0 = "mu_0"
    mu_1 = "mu_1"
    ite = "ite"
    rep = "rep"
    sample_id = "sample_id"


#: List of CausalFrame columns with defined meaning
DATA_COLS = [Col.t, Col.y, Col.y_cf, Col.y_0, Col.y_1, Col.mu_0, Col.mu_1, Col.ite]


class CausalFrame(pd.DataFrame):
    """Special DataFrame for causal data

    The CausalFrame ensures consistent naming of the columns in a DataFrame used
    for causal frames.

    """

    _metadata = ["_names"]

    def __init__(self, data, *args, **kwargs):
        covariates = kwargs.pop("covariates", None)
        internal_op = hasattr(self, "_names") or isinstance(data, BlockManager)

        super().__init__(data, *args, **kwargs)

        if not internal_op:
            # constructor called explicitly thus check parameters
            assert covariates is not None, "Parameter `covariates` missing"
            assert isinstance(covariates, (list, tuple)), "List of covariates needed"
            assert len(covariates) > 0, "At least one covariate column needed"
            assert set(covariates).issubset(
                set(self.columns)
            ), "Covariates must be a subset of columns"
            for col in DATA_COLS:
                assert col in self.columns, f"Column '{col}' not present!"

        self._names = dict(covariates=covariates)

    @property
    def _constructor(self) -> Type["CausalFrame"]:
        return CausalFrame

    @property
    def _constructor_sliced(self) -> Type[pd.Series]:
        # CausalSeries are not meaningful
        return pd.Series


@pd.api.extensions.register_dataframe_accessor("names")
class NamesAccessor:
    """Custom accessor to retrieve the names of covariates

    In order to access the covariate columns easily without having to inspect the
    DataFrame beforehand, this accessor allows to retrieve the covariate names

    Usage::

        >>> cf = load_ihdp(select_rep=0)[0]  # select replication 0f.names
        >>> cf.names.covariates
            ['0',
             '1',
             '2',
             '3',
             ...
             '21',
             '22',
             '23',
             '24']

    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(obj, CausalFrame), "CausalFrame is needed for this accessor"

    @property
    def covariates(self) -> List[str]:
        """Return covariate names of a CausalFrame"""
        return self._obj._names["covariates"]

    @property
    def treatment(self) -> str:
        """Return treatment indicator name of a CausalFrame"""
        return self._obj._names["treatment"]

    @property
    def outcome(self) -> str:
        """Return outcome name of a CausalFrame"""
        return self._obj._names["outcome"]

    @property
    def others(self) -> List[str]:
        """Return all other column names of a CausalFrame"""
        exclude = self._obj._names["covariates"] + DATA_COLS
        return [col for col in self._obj.columns if col not in exclude]


@pd.api.extensions.register_dataframe_accessor("np")
class NumpyAccessor:
    """Custom accessor to retrieve the numpy formatted columns

    Since `numpy` is the most important data format in the Python DataScience
    eco-system, this accessor allows to retrieve the numpy formatted data from a
    DataFrame by simply calling df.np

    Usage::

        >>> cf = load_ihdp(select_rep=[0])[0]
        >>> type(cf.np.X)
        <class 'numpy.ndarray'>
        >>> cf.np.X.shape
        (747, 25)


    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(obj, CausalFrame), "CausalFrame is needed for this accessor"

    @property
    def X(self) -> np.ndarray:
        """Return covariates as a numpy array"""
        cols = [col for col in self._obj.names.covariates if col in self._obj.columns]
        if not cols:
            raise KeyError("No known covariates in CausalFrame")
        return self._obj[cols].to_numpy()

    def __getattr__(self, col):
        """Return single column as numpy array"""
        return self._obj[col].to_numpy()
