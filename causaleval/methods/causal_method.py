
class DataInputError(Exception):
    pass

class CausalMethod():

    def __init__(self, seed=0):
        """

        :param seed: The random seed for any function using randomness
        """
        self.seed = seed
        self.is_fitted = False
        pass

    def fit(self, x, t, y, refit=False) -> None:
        """ Fit / Train the method on given data

        Must be called be before `predict_ite` or `predict_ate`

        :param refit: Whether to force a refitting on new data
        :param x: the covariate matrix with `k` columns and `n` rows
        :param t: the treatment vector with `n` elements
        :param y: the observed outcome vector `n` elements
        """
        pass

    def predict_ite(self, x):
        """ Predict individual treatment effect for given instances with
        covariates and treatment indicator
        :param x: covariates of instances
        :param t: treatment indicator of instances
        :return: an array of the individual treatment effects
        """
        pass

    def predict_ate(self, x):
        """ Predict average treatment effect for given population

        :param x: covariates of instances
        :param t: treatment indicator of instances
        :return: a float value of the ATE
        """
        pass

