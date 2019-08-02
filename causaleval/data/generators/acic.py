import pandas as pd
import numpy as np

from causaleval.data.data_provider import DataProvider
from causaleval import config

import scipy
from scipy import stats
from sklearn.preprocessing import StandardScaler

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
        covariates = pd.read_csv(config.IBM_PATH_ROOT + '/' + 'covariates.csv')
        covariates = covariates.drop(columns=['sample_id']).values


        def random_assignment(covariates):
            return np.random.random_integers(0,1, size=covariates.shape[0])

        def treatment_assignment(covariates):
            confounder_ids = np.random.choice(covariates.shape[1], size=10)
            confounders = covariates[:, confounder_ids]
            confounders = StandardScaler().fit_transform(confounders)

            expectations = scipy.special.expit(np.sum(confounders , axis=1)/len(confounder_ids))
            s = np.random.normal(expectations, scale=0.01)
            s = np.maximum(np.zeros(len(s)), s)
            s = np.minimum(np.ones(len(s)), s)
            return np.random.binomial(1, p=s)

        def treatment_effect(covariates):
            return np.full(len(covariates), 0.5)

        def outcome_assignment(covariates):
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





