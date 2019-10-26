import pandas as pd


class CausalSeries(pd.Series):
    @property
    def _constructor(self):
        return CausalSeries

    @property
    def _constructor_expanddim(self):
        return CausalFrame


class CausalFrame(pd.DataFrame):
    _metadata = ["_names"]

    def __init__(self, *args, **kwargs):
        covariates = kwargs.pop("covariates", [])
        treatment = kwargs.pop("treatment", "t")
        super().__init__(*args, **kwargs)
        self._names = dict(covariates=covariates, treatment=treatment)

    @property
    def _constructor(self):
        return CausalFrame

    @property
    def _constructor_sliced(self):
        return CausalSeries


@pd.api.extensions.register_dataframe_accessor("names")
class NamesAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(obj, CausalFrame), "Works only with CausalFrames"

    @property
    def covariates(self):
        # return the geographic center point of this DataFrame
        return self._obj._names["covariates"]

    @property
    def treatment(self):
        # return the geographic center point of this DataFrame
        return self._obj._names["treatment"]


@pd.api.extensions.register_dataframe_accessor("np")
class NumpyAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        assert isinstance(obj, CausalFrame), "Works only with CausalFrames"

    @property
    def X(self):
        cols = [col for col in self._obj.names.covariates if col in self._obj.columns]
        if not cols:
            raise IndexError("No known covariates in CausalFrame")
        return self._obj[cols].to_numpy()

    @property
    def t(self):
        if self._obj.names.treatment not in self._obj.columns:
            raise IndexError("No treatment variable in CausalFrame")
        else:
            return self._obj[self._obj.names.treatment]
