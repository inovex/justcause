"""
Specialised DataFrame which differentiates between covariates, treatment, outcome, etc.

More information under:
https://pandas.pydata.org/pandas-docs/stable/development/extending.html
"""
from __future__ import annotations

from functools import partial
from typing import List, Type

import numpy as np
import pandas as pd


class CausalFrame(pd.DataFrame):
    _metadata = ["_names"]

    def __init__(self, *args, **kwargs):
        covariates = kwargs.pop("covariates", None)
        treatment = kwargs.pop("treatment", "t")
        outcome = kwargs.pop("outcome", "y")
        internal_op = kwargs.pop("_internal_operation", False)

        assert covariates is not None, "Parameter `covariates` missing"
        super().__init__(*args, **kwargs)

        if not internal_op:
            assert isinstance(covariates, (list, tuple)), "List of covariates needed"
            assert len(covariates) > 0, "At least one covariate column needed"
            assert set(covariates).issubset(
                set(self.columns)
            ), "Covariates must be a subset of columns"
            assert treatment in self.columns, "Treatment must be a column name"
            assert outcome in self.columns, "Outcome must be a column name"

        self._names = dict(covariates=covariates, treatment=treatment, outcome=outcome)

    @property
    def _constructor(self) -> partial[CausalFrame]:
        # This is called during operations with CausalFrames
        # We pass a marker to handle cases when columns are lost
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
        assert isinstance(obj, CausalFrame), "Works only with CausalFrames"

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
            raise IndexError("No known covariates in CausalFrame")
        return self._obj[cols].to_numpy()

    @property
    def t(self) -> np.ndarray:
        """Return treatment as a numpy array"""
        if self._obj.names.treatment not in self._obj.columns:
            raise IndexError("No treatment variable in CausalFrame")
        else:
            return self._obj[self._obj.names.treatment].to_numpy()

    @property
    def y(self) -> np.ndarray:
        """Return outcome as a numpy array"""
        if self._obj.names.outcome not in self._obj.columns:
            raise IndexError("No outcome variable in CausalFrame")
        else:
            return self._obj[self._obj.names.outcome].to_numpy()
