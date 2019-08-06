import os

import numpy as np
import pandas as pd

from causaleval.data.data_provider import DataProvider
from causaleval import config

from itertools import cycle

class SimpleIBMDataProvider(DataProvider):
    """
    Simply return one of 50k sized datasets for now
    """

    def __init__(self):
        super().__init__()
        self.x = None
        self.t = None
        self.y = None
        self.y_cf = None

    def get_test_data(self):
        super().get_test_data()

    def get_training_data(self, size=None):
        if self.x is None:
            self.load_training_data()
        if size is None:
            return self.x, self.t, self.y

    def __str__(self):
        return "IBM"

    def get_true_ite(self, data=None):
        return self.treated_outcome - self.control_outcome

    def get_true_ate(self, subset=None):
        return np.mean(self.get_true_ite())

    def load_training_data(self):
        params = pd.read_csv(config.IBM_PATH + '/' + 'params.csv')
        id = params[params['size'] == 50000]['ufid'].iloc[0]

        factual = pd.read_csv(config.IBM_PATH + '/' + 'factuals'+ '/' + id + '.csv')
        counterfactual = pd.read_csv(config.IBM_PATH + '/' + 'counterfactuals'+ '/' + id + '_cf.csv')
        covariates = pd.read_csv(config.IBM_PATH_ROOT + '/' + 'covariates.csv')
        self.x = covariates[covariates['sample_id'].isin(list(factual['sample_id']))].drop(columns=['sample_id']).values
        self.y = factual['y'].values
        self.t = factual['z'].values
        union = counterfactual[['y0','y1']].values

        # Retrieve counterfactuals
        cf = []
        i = 0
        for t in self.t:
            cf.append(union[i, 1-t])
            i += 1

        self.params = params[params['ufid'] == id]
        self.y_cf = np.array(cf)
        self.treated_outcome = counterfactual['y1']
        self.control_outcome = counterfactual['y0']


    def get_params(self):
        return self.params







