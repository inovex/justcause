"""
Generalised DataFrame which differentiates between covariates and other column names
"""
from __future__ import annotations

from abc import ABC
from functools import partial
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


class CausalFrame(pd.DataFrame, ABC):
    _metadata = ["_names"]

    def __init__(self, data, *args, **kwargs):
        covariates = kwargs.pop("covariates", None)
        internal_op = kwargs.pop("_internal_operation", False) or isinstance(
            data, BlockManager
        )

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
    def _constructor(self) -> partial[CausalFrame]:
        # This is called during operations with CausalFrames
        # We pass a marker to differentiate between explicit and implicit invocation
        kwargs = {"_internal_operation": True, **self._names}
        return partial(CausalFrame, **kwargs)

    @property
    def _constructor_sliced(self) -> Type[pd.Series]:
        # CausalSeries are not meaningful
        return pd.Series


@pd.api.extensions.register_dataframe_accessor("names")
class NamesAccessor:
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
