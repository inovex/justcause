import numpy as np


class CausalForest:
    """ Port for the R implementation of CausalForests using rpy2

    References:
        [1] “Generalized random forests”
            S. Athey, J. Tibshirani, and S. Wager,
            Ann. Stat., vol. 47, no. 2, pp. 1179–1203, 2019.
    """

    def __init__(self, random_state: int = 0):
        """ Checks if the required R package is available and instantiates it

        We don't install the grf package automatically, because it takes to long with
        the required compilation. It would interrupt application workflow and is not
        the task of our library.
        The needed method for installation can be found in `justcause.learners.utils`

        Args:
            random_state: random seed that is passed to the R implementation
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

        assert type(random_state) is int, (
            "Only integer type random state " "can be passed to rpy2"
        )

        """ Holds the rpy2 object for the trained model"""
        self.forest = None
        self.random_state = random_state

    def __str__(self):
        return "CausalForest"

    def predict_ate(self, x, t=None, y=None):
        """ Predict ATE for the given samples"""
        return np.mean(self.predict_ite(x))

    def predict_ite(self, x, t=None, y=None):
        """ Predicts ITEs for given samples"""
        import rpy2.robjects as robjects

        pred = robjects.r.predict(self.forest, x, estimate_variance=False)[0]
        return np.array(pred).flatten()

    def fit(self, x, t, y):
        """ Fits the forest using factual data"""
        from rpy2.robjects.vectors import FloatVector, IntVector

        self.forest = self.grf.causal_forest(
            x, FloatVector(y), IntVector(t), seed=self.random_state
        )
