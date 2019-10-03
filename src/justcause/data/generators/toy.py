import config
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale

from ..data_provider import DataProvider


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def exponential_effect(x):
    return np.exp(1 + x[:, 4] - x[:, 5] / 2)  # use birth weight


def multi_effect(x):
    effect = (x[:, 1] * x[:, 2]) ** 3 + x[:, 4] * x[:, 5] ** 2
    return minmax_scale(effect, feature_range=(1, 10))


def multi_modal_effect(X, conf_idx=6):
    prob = sigmoid(X[:, conf_idx]) > 0.5
    return np.random.normal(
        (3 * prob) + 1 * (1 - prob), 0.1, size=len(X)
    )  # Explicitly multimodal


def exponential_effect_old(X, conf_idx=6):
    return np.exp(1 + sigmoid(X[:, conf_idx]))  # use birth weight


class SWagerDataProvider(DataProvider):
    """
    Implements the toy examples proposed by S. Wager in a personal communication
    to show the efficacy of Causal Forests
    """

    def __init__(self, seed=0, setting="simple"):
        self.setting = setting
        super().__init__(seed, train_size=0.8)

    def __str__(self):
        return "SWager-Toy" + self.setting

    def load_training_data(self):

        n = 2000
        p = 7
        noise_scale = 0

        rand = np.random.normal(0, 1, size=n * p)
        X = rand.reshape((n, p))
        rand = np.random.normal(0, 1, size=n * p)  # new random
        X_test = rand.reshape((n, p))

        noise = np.random.normal(scale=noise_scale, size=n)  # add some noise

        if self.setting == "simple":
            # Big treatment effect, no confounding
            ite = sigmoid(X[:, 2] + X[:, 3]) * 3  # make effect large, but all positive
            T = np.random.binomial(1, 0.5, size=n)  # random assignment
            Y = sigmoid(X[:, 1]) + (ite * T)
            Y_cf = sigmoid(X[:, 1]) + (ite * (1 - T))
        elif self.setting == "hard":
            # Small treatment effect, a little confounding
            ite = sigmoid(X[:, 2] + X[:, 3]) / 2
            T = np.random.binomial(1, sigmoid(X[:, 1]), size=n)  # non-random assignment
            Y = sigmoid(X[:, 1]) + ite * T
            Y_cf = sigmoid(X[:, 1]) + ite * (1 - T)
        elif self.setting == "small_rct":
            # Small treatment effect, no confounding
            ite = sigmoid(X[:, 2] + X[:, 3]) / 2
            ite_test = sigmoid(X_test[:, 2] + X_test[:, 3]) / 2
            T = np.random.binomial(1, 0.5, size=n)  # random assignment
            Y = sigmoid(X[:, 1]) + ite * T
            Y_cf = sigmoid(X[:, 1]) + ite * (1 - T)
        elif self.setting == "more_conf":
            # confounded treatment effect, large.
            ite = sigmoid(X[:, 2] + X[:, 3]) * 3
            ite_test = sigmoid(X_test[:, 2] + X_test[:, 3]) / 2
            T = np.random.binomial(
                1, sigmoid(X[:, 2] + X[:, 3]), size=n
            )  # people with higher effect get treatment
            Y = sigmoid(X[:, 1]) + ite * T
            Y_cf = sigmoid(X[:, 1]) + ite * (1 - T)
        elif self.setting == "more_conf_small":
            # confounded treatment effect, small.
            ite = sigmoid(X[:, 2] + X[:, 3])
            ite_test = sigmoid(X_test[:, 2] + X_test[:, 3]) / 2  # noqa: F841
            T = np.random.binomial(
                1, sigmoid(X[:, 2] + X[:, 3]), size=n
            )  # people with higher effect get treatment
            Y = sigmoid(X[:, 1]) + ite * T
            Y_cf = sigmoid(X[:, 1]) + ite * (1 - T)

        self.x = X
        self.y = Y
        self.y_cf = Y_cf
        self.y_0 = sigmoid(X[:, 1]) + noise
        self.y_1 = self.y_0 + ite
        self.t = T
        self.x_test = X_test


class SWagerRealCompare(DataProvider):
    def __init__(self, setting="simple"):
        """
        :param params: dict containing 'random', 'homogeneous', 'deterministic',
                       'confounded'
        """
        self.setting = setting
        super().__init__()

    def __str__(self):
        return "SWager-Real" + self.setting

    def load_training_data(self):
        covariates_df = pd.read_csv(config.IBM_PATH_ROOT + "/" + "covariates.csv")
        covariates_df = covariates_df.loc[:, covariates_df.var() > 0.3]
        self.covariates_df = covariates_df.drop(columns=["sample_id"])
        self.covariates = covariates_df[config.ACIC_USE_COVARIATES].values
        self.x = self.covariates

        n = len(self.x)
        p = self.x.shape[1]  # noqa: F841
        X = StandardScaler().fit_transform(self.x)
        self.x = X
        noise_scale = 0
        noise = np.random.normal(scale=noise_scale, size=n)  # add some noise

        if self.setting == "simple":
            # Big treatment effect, no confounding
            ite = sigmoid(X[:, 2] + X[:, 3]) * 2  # make effect large, but all positive
            T = np.random.binomial(1, 0.5, size=n)  # random assignment
            Y = sigmoid(X[:, 1]) + (ite * T) + noise
            Y_cf = sigmoid(X[:, 1]) + (ite * (1 - T)) + noise
        elif self.setting == "hard":
            # Small treatment effect, a little confounding
            ite = exponential_effect(X)
            T = np.random.binomial(
                1, sigmoid((X[:, 2] + X[:, 3]) / 2), size=n
            )  # non-random assignment
            Y = sigmoid(X[:, 1]) + ite * T + noise
            Y_cf = sigmoid(X[:, 1]) + ite * (1 - T) + noise

        self.x = X
        self.y = Y
        self.y_cf = Y_cf
        self.y_0 = sigmoid(X[:, 1]) + noise
        self.y_1 = self.y_0 + ite
        self.t = T


class Second(DataProvider):
    def __init__(self, setting="simple"):
        """
        :param params: dict containing 'random', 'homogeneous', 'deterministic',
                       'confounded'
        """
        self.setting = setting
        super().__init__()

    def __str__(self):
        return "New-Experiment" + self.setting

    def load_training_data(self):
        covariates_df = pd.read_csv(config.IBM_PATH_ROOT + "/" + "covariates.csv")
        covariates_df = covariates_df.loc[:, covariates_df.var() > 0.3]
        self.covariates_df = covariates_df.drop(columns=["sample_id"])
        self.covariates = covariates_df[config.ACIC_USE_COVARIATES].values
        self.x = self.covariates

        n = len(self.x)
        p = self.x.shape[1]  # noqa: F841
        X = StandardScaler().fit_transform(self.x)
        self.x = X
        noise_scale = 0.1
        noise = np.random.normal(scale=noise_scale, size=n)  # add some noise

        if self.setting == "single":
            # Strong confounding based on one covariate -> mean 0
            ite = X[:, 2]  # make effect large, but all positive
            T = np.random.binomial(
                1, minmax_scale(X[:, 2].reshape(-1, 1), feature_range=(0, 1))[0], size=n
            )  # random assignment
            Y = sigmoid(X[:, 1]) + (ite * T) + noise
            Y_cf = sigmoid(X[:, 1]) + (ite * (1 - T)) + noise
        elif self.setting == "multi":
            # Confounding based on interaction of many covariates
            ite = multi_effect(X)
            T = np.random.binomial(
                1, minmax_scale(ite.reshape(-1, 1), feature_range=(0, 1))[0], size=n
            )  # non-random assignment
            Y = sigmoid(X[:, 1]) + ite * T + noise
            Y_cf = sigmoid(X[:, 1]) + ite * (1 - T) + noise

        self.x = X
        self.y = Y
        self.y_cf = Y_cf
        self.y_0 = sigmoid(X[:, 1]) + noise
        self.y_1 = self.y_0 + ite
        self.t = T


class MultiExpoACICGenerator(DataProvider):
    def __init__(self, setting="multi-modal"):
        """
        :param params: dict containing 'random', 'homogeneous', 'deterministic',
                       'confounded'
        """
        self.setting = setting
        super().__init__()

    def __str__(self):
        return "ACIC" + self.setting

    def load_training_data(self):
        covariates_df = pd.read_csv(config.IBM_PATH_ROOT + "/" + "covariates.csv")
        covariates_df = covariates_df.loc[:, covariates_df.var() > 0.3]
        self.covariates_df = covariates_df.drop(columns=["sample_id"])
        self.covariates = covariates_df[config.ACIC_USE_COVARIATES].values
        self.x = self.covariates

        conf_idx = 2

        n = len(self.x)
        p = self.x.shape[1]  # noqa: F841
        X = StandardScaler().fit_transform(self.x)
        self.x = X[X[:, 2] < 8]
        X = self.x
        noise_scale = 0.1
        # add some noise
        noise = np.random.normal(scale=noise_scale, size=n)  # noqa: F841

        Y_0 = np.random.gamma(0.2, 1, size=len(X))

        if self.setting == "multi-modal":
            Y_1 = Y_0 + multi_modal_effect(X, conf_idx)
            val = X[:, conf_idx]
        else:
            Y_1 = Y_0 + exponential_effect_old(X, conf_idx)
            val = X[:, conf_idx]

        self.y_0 = Y_0
        self.y_1 = Y_1
        self.t = np.random.binomial(1, p=sigmoid(val), size=len(X))
        union = np.c_[self.y_0, self.y_1]
        self.y_cf = np.array([row[int(1 - ix)] for row, ix in zip(union, self.t)])
        self.y = np.array([row[int(ix)] for row, ix in zip(union, self.t)])
        self.x = X


class CovariateModulator(DataProvider):
    def __init__(self, setting="big"):
        """
        :param params: dict containing 'random', 'homogeneous', 'deterministic',
                       'confounded'
        """
        self.setting = setting
        super().__init__()

    def __str__(self):
        return "CovariateModulator" + self.setting

    def load_training_data(self):
        covariates_df = pd.read_csv(config.IBM_PATH_ROOT + "/" + "covariates.csv")
        covariates_df = covariates_df.loc[:, covariates_df.var() > 0.3]
        if self.setting == "big":
            self.covariates_df = covariates_df.drop(columns=["sample_id"])
            self.covariates = covariates_df.values
            self.x = self.covariates
            self.idxs = [
                self.covariates_df.columns.get_loc(c) + 1
                for c in config.ACIC_USE_COVARIATES
            ]
        else:
            self.covariates_df = covariates_df.drop(columns=["sample_id"])
            self.covariates = covariates_df[config.ACIC_USE_COVARIATES].values
            self.x = self.covariates

        n = len(self.x)
        X = StandardScaler().fit_transform(self.x)
        self.x = X

        if self.setting == "big":
            ite = sigmoid(X[:, self.idxs[2]] + X[:, self.idxs[3]]) / 2
            T = np.random.binomial(
                1, sigmoid(X[:, self.idxs[2]] + X[:, self.idxs[3]]) / 2, size=n
            )  # non-random assignment
            Y_0 = sigmoid(X[:, self.idxs[4]])
            Y_1 = Y_0 + ite
            Y = Y_0 + ite * T
            Y_cf = Y_0 + ite * (1 - T)
        else:
            # Small treatment effect, a little confounding
            ite = sigmoid(X[:, 2] + X[:, 3]) / 2
            T = np.random.binomial(
                1, sigmoid(X[:, 2] + X[:, 3]) / 2, size=n
            )  # non-random assignment
            Y_0 = sigmoid(X[:, 4])
            Y_1 = Y_0 + ite
            Y = Y_0 + ite * T
            Y_cf = Y_0 + ite * (1 - T)

        self.x = X
        self.y = Y
        self.y_cf = Y_cf
        self.y_0 = Y_0
        self.y_1 = Y_1
        self.t = T
        self.true_train_ite = ite


# os.environ['R_HOME'] = config.R_HOME


if __name__ == "__main__":

    import utils

    gen = MultiExpoACICGenerator(setting="multi-modal")
    utils.confounder_outcome_plot(
        gen.x[:, 2], gen.y_1 - gen.y_0, dataset="ACIC-multi-modal"
    )
    utils.plot_y_dist(gen.y, gen.y_cf, method_name="ACIC-multi-modal")
    gen = MultiExpoACICGenerator(setting="exponential")
    utils.confounder_outcome_plot(
        gen.x[:, 2], gen.y_1 - gen.y_0, dataset="ACIC-exponential"
    )
    utils.plot_y_dist(gen.y, gen.y_cf, method_name="ACIC-exponential")
