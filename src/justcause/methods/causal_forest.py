from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector, FloatVector, IntVector
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

import numpy as np

from .causal_method import CausalMethod


class CausalForest(CausalMethod):

    def __init__(self, seed=0):
        super().__init__(seed)
        self.grf = self.install_grf()
        self.forest = None

    def __str__(self):
        return "Causal Forest"

    @staticmethod
    def install_grf():
        """Install the `grf` R package and active necessary conversion

        :return: The robject for `grf`
        """

        # robjects.r is a singleton
        robjects.r.options(download_file_method='curl')
        numpy2ri.activate()
        package_names = ["grf"]
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=0)

        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))

        return importr("grf")

    def predict_ate(self, x, t=None, y=None):
        predictions = self.predict_ite(x)
        return np.mean(predictions)

    def predict_ite(self, x, t=None, y=None):
        if self.forest is None:
            raise AssertionError('Must fit the forest before prediction')

        pred = robjects.r.predict(self.forest, x, estimate_variance=False)
        return np.array(list(map(lambda element: element[0], pred)))

    def fit(self, x, t, y, refit=False):
        print('fit forest anew')
        self.forest = self.grf.causal_forest(x, FloatVector(y), IntVector(t), seed=self.seed)

