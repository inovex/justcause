import pandas as pd
import numpy as np

from causaleval.data.data_provider import DataProvider
from causaleval import config

import scipy
from sklearn.preprocessing import StandardScaler

# Playground imports
import seaborn as sns
import matplotlib
from sklearn.ensemble import RandomForestRegressor
from causaleval.data.sets.ibm import SimpleIBMDataProvider


# To make it work on MacOS
import matplotlib
matplotlib.use("MacOSX")

import seaborn as sns
sns.set(style="darkgrid")

import matplotlib.pyplot as plt


class ACICGenerator(DataProvider):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ACIC"

    def get_true_ite(self, data=None):
        super().get_true_ite(data)

    def get_train_generator_batch(self, batch_size=32):
        pass

    def get_training_data(self, size=None):
        super().get_training_data(size)

    def get_train_generator_single(self, random=False, replacement=False):
        pass

    def get_true_ate(self, subset=None):
        super().get_true_ate(subset)

    def get_num_covariates(self):
        pass

    def get_info(self):
        super().get_info()

    def get_test_data(self):
        super().get_test_data()

    def generate_data(self):
        np.random.seed(0)
        covariates_df = pd.read_csv(config.IBM_PATH_ROOT + '/' + 'covariates.csv')
        sub_df = covariates_df.loc[:,[x.startswith('f_') for x in covariates_df.columns]]
        covariates = covariates_df.drop(columns=['sample_id']).values

        # rf = RandomForestRegressor(n_jobs=-1)
        # x, t, y = SimpleIBMDataProvider().get_training_data()
        # rf.fit(X=x, y=y)
        # print('outcome importance')
        # print(np.flip(np.argsort(rf.feature_importances_)))
        #
        # rf.fit(X=x, y=t)
        # print('treatment importance')
        # print(np.flip(np.argsort(rf.feature_importances_)))

        def normal_polynomial(vars):
            """
            Calculate the value of linear function with normal sampled coefficients for the input vars
            :param vars:
            :return:
            """
            coeffs = np.random.randn(len(vars))
            mult = coeffs*vars
            return np.sum(mult)/len(vars)

        def random_assignment(covariates):
            return np.random.random_integers(0,1, size=covariates.shape[0])

        def treatment_assignment(covariates, num_parents=10, equal_split=True, relation='weak', use_parents=None):
            if relation == 'random':
                return random_assignment(covariates)
            if relation == 'weak':
                if use_parents is not None:
                    confounders = use_parents
                else:
                    ids = np.random.choice(covariates.shape[1], size=num_parents)
                    confounders = StandardScaler().fit_transform(covariates[:, ids])

                exp_poly = scipy.special.expit(np.array(list(map(normal_polynomial, confounders))))
                return np.random.binomial(1, p=exp_poly)
            if relation == 'strong':
                # Idea: Create a deterministic split based on a few covariates that results in a 50/50
                # partition of the data
                pass

        def treatment_effect(covariates, homogeneous=True):
            if homogeneous:
                return np.full(len(covariates), 0.5)
            else:
                return 0 # TODO

                

        def outcome_assignment(covariates, num_parents=10, equal_split=True, relation='weak', use_parents=None):
            standardized = StandardScaler().fit_transform(covariates)
            y_0 = (standardized[:, 0]**2 + standardized[:, 1]**(1/2))*3
            y_1 = y_0 + treatment_effect(covariates)
            return np.concatenate(y_1, y_0, axis=1)

        t_random = random_assignment(covariates)
        t = treatment_assignment(covariates)
        print(t)

        ys = outcome_assignment(covariates)

        print('here')







if __name__ == '__main__':
    a = ACICGenerator()
    a.generate_data()





