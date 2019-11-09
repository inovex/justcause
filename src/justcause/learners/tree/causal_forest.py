from typing import Optional, Union

import numpy as np
from numpy.random import RandomState
from sklearn.utils import check_random_state

from ...utils import int_from_random_state

#: type alias
StateType = Optional[Union[int, RandomState]]


class CausalForest:
    """ Port for the R implementation of CausalForests using rpy2

    See reference [2] for a list of parameters to the original implementation.
    In general, points are converted to underscores. Thus `num.trees` becomes
    `num_trees` when using the Rpy2 API


    References:
        [1] “Generalized random forests”
            S. Athey, J. Tibshirani, and S. Wager,
            Ann. Stat., vol. 47, no. 2, pp. 1179–1203, 2019.

        [2] The CRAN manual for `grf`
            https://cran.r-project.org/web/packages/grf/grf.pdf
    """

    def __init__(
        self, num_trees: int = 200, random_state: RandomState = None, **kwargs
    ):
        """ Checks if the required R package is available and instantiates it

        We don't install the grf package automatically, because it takes to long with
        the required compilation. It would interrupt application workflow and is not
        the task of our library.
        The needed method for installation can be found in `justcause.learners.utils`

        Args:
            random_state: random seed that is passed to the R implementation
            num_trees: the number of trees in the forest
            kwargs: named parameters that are passed to the R implementation,
                see https://cran.r-project.org/web/packages/grf/grf.pdf for a reference
                of possible parameters
        """
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        from rpy2.rinterface import RRuntimeError

        numpy2ri.activate()

        try:
            self.grf = importr("grf")
        except RRuntimeError:
            raise ImportError(
                "R package 'grf' is not installed yet, "
                "install it with justcause.learners.utils.install_r_packages(['grf'])"
            )

        """ Holds the rpy2 object for the trained model"""
        self.forest = None
        self.random_state = check_random_state(random_state)

        self.num_trees = num_trees
        self.kwargs = kwargs

    def __str__(self):
        return "CausalForest"

    def estimate_ate(self, x: np.array, t: np.array, y: np.array) -> float:
        """ Estimates ATE of the given population

        Fits the CausalForest and predicts the ITE. The mean of all ITEs is
        returned as the ATE.

        Args:
            x: covariates
            t: treatment indicator
            y: factual outcome

        Returns: average treatment effect of the population
        """
        self.fit(x, t, y)
        ite = self.predict_ite(x)
        return float(np.mean(ite))

    def predict_ite(self, x: np.array, t=None, y=None) -> np.array:
        """
        Predicts ITE vor given samples without using facutals

        Args:
            x:
            t:  NOT USED - kept for coherent API
            y:  NOT USED - kept for coherent API

        Returns:

        """
        import rpy2.robjects as robjects

        pred = robjects.r.predict(self.forest, x, estimate_variance=False)[0]
        return np.array(pred).flatten()

    def fit(self, x: np.array, t: np.array, y: np.array) -> None:
        """ Fits the forest using factual data"""
        from rpy2.robjects.vectors import FloatVector, IntVector

        integer_random_state = int_from_random_state(self.random_state)

        self.forest = self.grf.causal_forest(
            x,
            FloatVector(y),
            IntVector(t),
            seed=integer_random_state,
            num_trees=self.num_trees,
            **self.kwargs
        )
