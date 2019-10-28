"""
ToDo: Keep this a pure wrapper but don't just install the R packages.
      Rather provide this in a class method function.
ToDo: Check if this could  be done with the sklearn random forest too?
"""

import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from learners.causal_method import CausalMethod
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, IntVector, StrVector


class CausalForest(CausalMethod):
    def __init__(self, seed=0):
        super().__init__(seed)
        # ToDo: Check if this should rather be done only once somewhere in __init__.py
        numpy2ri.activate()
        self.grf = importr("grf")
        self.forest = None

    def __str__(self):
        return "Causal Forest"

    @staticmethod
    def install_grf():
        """Install the `grf` R package and active necessary conversion

        :return: The robject for `grf`

        ToDo: Make this a function that can download several things, not only "grf"
        """
        # robjects.r is a singleton
        robjects.r.options(download_file_method="curl")
        numpy2ri.activate()
        package_names = ["grf"]
        utils = rpackages.importr("utils")
        # ToDo: No user interaction when running functions form a library
        # utils.chooseCRANmirror(ind=0)

        names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
        if len(names_to_install) > 0:
            # Todo: Expect the user to have set a proper repo and check it.
            #       This is only a workaround
            utils.install_packages(
                StrVector(names_to_install), repos="http://cran.us.r-project.org"
            )

    def predict_ate(self, x, t=None, y=None):
        predictions = self.predict_ite(x)
        return np.mean(predictions)

    def predict_ite(self, x, t=None, y=None):
        if self.forest is None:
            raise AssertionError("Must fit the forest before prediction")

        pred = robjects.r.predict(self.forest, x, estimate_variance=False)
        return np.array(list(map(lambda element: element[0], pred)))

    def fit(self, x, t, y, refit=False):
        print("fit forest anew")
        self.forest = self.grf.causal_forest(
            x, FloatVector(y), IntVector(t), seed=self.seed
        )
