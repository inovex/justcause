import numpy as np

from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector, FloatVector, IntVector
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

from ..causal_method import CausalMethod


class Bart(CausalMethod):

    def __init__(self, seed=0):
        super().__init__(seed)
        self.bm = self.install_bart()
        self.model = None

    def __str__(self):
        return "BART - "

    @staticmethod
    def install_bart():
        """Install the `grf` R package and active necessary conversion

        :return: The robject for `grf`
        """

        # robjects.r is a singleton
        robjects.r.options(download_file_method='curl')
        print(robjects.r("version"))
        numpy2ri.activate()
        package_names = ["BayesTree"]
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)

        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))

        return importr("BayesTree")

    @staticmethod
    def union(x, t):
        return np.c_[x, t]

    def predict_ite(self, x, t=None, y=None):
        treated = self.union(x, np.ones(x.shape[0]))
        control = self.union(x, np.zeros(x.shape[0]))
        pred1 = robjects.r.predict(self.model, treated)
        pred0 = robjects.r.predict(self.model, control)
        return pred1 - pred0

    def fit(self, x, t, y, refit=False) -> None:
        train = self.union(x, t)
        self.model = robjects.r.bart(train, FloatVector(y))


