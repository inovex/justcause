from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector, FloatVector, IntVector
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

import numpy as np

from .causal_method import CausalMethod


class RLearner(CausalMethod):
    """
    Uses the R package provided by X.Nie and S. Wager in ﻿https://arxiv.org/pdf/1712.04912.pdf
    """

    def __init__(self, seed=0, method='lasso'):
        super().__init__()
        self.rleaner = self.install_rlearner()
        self.model = None
        self.method_name = method

    def __str__(self):
        return "R-Learner-"+self.method_name.capitalize()

    @staticmethod
    def install_rlearner():
        """Load the `rlearner` R package and activate necessary conversion

        :return: The robject for `rlearner`
        """

        # robjects.r is a singleton
        robjects.r.options(download_file_method='curl')
        numpy2ri.activate()
        package_names = ["devtools"]
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=0)

        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))

        return importr('rlearner')

    def predict_ate(self, x, t=None, y=None):
        predictions = self.predict_ite(x)
        return np.mean(predictions)

    def predict_ite(self, x, t=None, y=None):
        if self.model is None:
            raise AssertionError('Must fit the forest before prediction')

        return np.array(robjects.r.predict(self.model, x)).reshape(1, -1)[0]

    def fit(self, x, t, y, refit=False):
        if self.method_name == 'lasso':
            print('fit lasso')
            self.model = self.rleaner.rlasso(x, IntVector(t), FloatVector(y))
        else:
            # Takes much longer to fit
            print('fit boost')
            self.model = self.rleaner.rboost(x, IntVector(t), FloatVector(y))


class XLearner(CausalMethod):
    """
    Uses the R package provided by X.Nie and S. Wager in ﻿https://arxiv.org/pdf/1712.04912.pdf
    """
    def __init__(self, seed=0, method='lasso'):
        super().__init__()
        self.rleaner = self.install_rlearner()
        self.model = None
        self.method_name = method

    def __str__(self):
        return "X-Learner-"+self.method_name.capitalize()

    @staticmethod
    def install_rlearner():
        """Load the `rlearner` R package and activate necessary conversion

        :return: The robject for `rlearner`
        """

        # robjects.r is a singleton
        robjects.r.options(download_file_method='curl')
        numpy2ri.activate()
        package_names = ["devtools"]
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=0)

        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            utils.install_packages(StrVector(names_to_install))

        return importr('rlearner')

    def predict_ate(self, x, t=None, y=None):
        predictions = self.predict_ite(x)
        return np.mean(predictions)

    def predict_ite(self, x, t=None, y=None):
        if self.model is None:
            raise AssertionError('Must fit the forest before prediction')

        return np.array(robjects.r.predict(self.model, x)).reshape(1, -1)[0]

    def fit(self, x, t, y, refit=False):
        if self.method_name == 'lasso':
            print('fit lasso')
            self.model = self.rleaner.xlasso(x, IntVector(t), FloatVector(y))
        else:
            # Takes much longer to fit
            print('fit boost')
            self.model = self.rleaner.xboost(x, IntVector(t), FloatVector(y))


