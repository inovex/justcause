import os

import numpy as np
import pandas as pd

from ..data_provider import DataProvider

import config


class SimpleIBMDataProvider(DataProvider):
    """
    Simply return one of 50k sized datasets for now
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "IBM"

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
        self.y_1 = counterfactual['y1']
        self.y_0 = counterfactual['y0']


    def get_params(self):
        return self.params







