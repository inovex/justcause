
class DataInputError(Exception):
    pass


# Todo: Make this a proper abstract class
class CausalMethod:

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

    def predict_ite(self, x, t=None, y=None):
        """Predict individual treatment effect for given instances with
        covariates and treatment indicator
        :param x: covariates of instances
        :param t: treatment indicator of instances
        :param y: outcome of instances
        :return: an array of the individual treatment effects
        """
        pass

    def predict_ate(self, x, t=None, y=None):
        """Predict average treatment effect for given population

        :param x: covariates of instances
        :param t: treatment indicator of instances
        :param y: outcome of instances
        :return: a float value of the ATE
        """
        pass

    def __str__(self):
        """
        Overwrite the string representation to get a method identifier for the logs
        :return:
        """
        return "Abstract Causal Method"

    def requires_provider(self):
        return False

    def fit_provider(self, data_provider):
        raise NotImplementedError('No provider based training implemented')
