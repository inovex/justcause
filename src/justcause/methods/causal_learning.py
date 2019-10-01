"""
All methods provided in https://github.com/saberpowers/causalLearning
introduced in the paper: ﻿https://arxiv.org/pdf/1707.00102.pdf
"""
import os
import config
os.environ['L_ALL'] = 'en_US.UTF-8'
os.environ['R_HOME'] = config.R_HOME


from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector, FloatVector, IntVector
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

import numpy as np

from methods.causal_method import CausalMethod

def install_cl():
    """Load the `causalLearning` R package and activate necessary conversion

    :return: The robject for `causalLearning`
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

    return importr('causalLearning')

class CausalLearningMethod(CausalMethod):
    """ABSTRACT"""

    def __init__(self, seed=0):
        super().__init__()
        self.cl = install_cl()
        self.model = None

    def predict_ite(self, x, t=None, y=None):
        if self.model is None:
            raise AssertionError('Must fit the method before prediction')

        return np.array(robjects.r.predict(self.model, x)).reshape(1, -1)[0]


class CausalBoosting(CausalLearningMethod):

    def __init__(self, seed=0):
        super().__init__()

    def __str__(self):
        return "CausalBoosting"

    def fit(self, x, t, y, refit=False):
        print('fit causal boost')
        self.model = self.cl.causalBoosting(x, IntVector(t), FloatVector(y), num_trees=500)

    def predict_ite(self, x, t=None, y=None):
        if self.model is None:
            raise AssertionError('Must fit the method before prediction')

        pred = np.array(robjects.r.predict(self.model, x, t, type='treatment.effect'))

        return np.mean(pred, axis=1)


class PolinatedTransformedOutcomeForest(CausalLearningMethod):
    def __init__(self, seed=0):
        super().__init__()

    def __str__(self):
        return "PTOforest"

    def fit(self, x, t, y, refit=False):
        print('fit PTOForest')
        self.model = self.cl.PTOforest(x, IntVector(t), FloatVector(y))


class CausalMars(CausalLearningMethod):

    def __init__(self, seed=0):
        super().__init__()
        self.cl = install_cl()
        self.model = None

    def __str__(self):
        return "CausalMARS"

    def fit(self, x, t, y, refit=False):
        print('fit causalMARS')
        self.model = self.cl.causalMARS(x, IntVector(t), FloatVector(y))

if __name__ == '__main__':

    from data.sets.ihdp import IHDPCfrProvider
    ihdp = IHDPCfrProvider()
    cb = CausalBoosting()
    cb.fit(*ihdp.get_training_data())
    cb.predict_ite(*ihdp.get_test_data())