import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, IntVector

# Activate parsing once CausalForest module is accessed
numpy2ri.activate()


class CausalForest:
    def __init__(self, seed=0):
        try:
            self.grf = importr("grf")
        except:
            raise ModuleNotFoundError(
                "R package grf is not installed yet, "
                "install it with utils.install_r_packages"
            )

        self.forest = None
        self.seed = seed

    def __str__(self):
        return "CausalForest"

    def predict_ate(self, x, t=None, y=None):
        return np.mean(self.predict_ite(x))

    def predict_ite(self, x, t=None, y=None):
        if self.forest is None:
            raise AssertionError("Must fit the forest before prediction")

        pred = robjects.r.predict(self.forest, x, estimate_variance=False)
        return np.array(list(map(lambda element: element[0], pred)))

    def fit(self, x, t, y):
        print("fit forest anew")
        self.forest = self.grf.causal_forest(
            x, FloatVector(y), IntVector(t), seed=self.seed
        )
