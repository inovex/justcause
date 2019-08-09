# Dynamically set the environment variable for R_HOME as
# found by running python -m rpy2.situation from the terminal
import os
import time

import pandas as pd

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

# Sacred
from sacred import Experiment
from sacred.observers import MongoObserver

# Base Methods
from sklearn.ensemble.forest import RandomForestRegressor, DecisionTreeRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


from causaleval.metrics import EvaluationMetric, StandardEvaluation

# Methods
from causaleval.methods.causal_method import CausalMethod
from causaleval.methods.basics.outcome_regression import SingleOutcomeRegression, DoubleOutcomeRegression
from causaleval.methods.basics.double_robust import DoubleRobust
from causaleval.methods.basics.propensity_weighting import PropensityScoreWeighting
from causaleval.methods.causal_forest import CausalForest
from causaleval.methods.ganite_wrapper import GANITEWrapper
from causaleval.methods.dragonnet_wrapper import DragonNetWrapper

# Data
from causaleval.data.generators.acic import ACICGenerator
from causaleval.data.sets.ihdp import IHDPDataProvider
from causaleval.data.sets.twins import TwinsDataProvider
from causaleval.data.sets.ibm import SimpleIBMDataProvider
from causaleval import config

ex = Experiment('normal')
ex.observers.append(MongoObserver.create(url=config.DB_URL, db_name=config.DB_NAME))

def create_data_gens():
    tf_list = [True, False]
    data_gen_list = []

    for one in tf_list:
        for two in tf_list:
            dict = {
                'random' : one,
                'homogeneous' : two,
                'deterministic': False,
                'confounded' : False,
                'seed' : 0
            }
            data_gen_list.append(ACICGenerator(dict))

    conf_dict = {
                'random' : False,
                'homogeneous' : False,
                'deterministic': False,
                'confounded' : True,
                'seed' : 0
            }
    data_gen_list.append(ACICGenerator(conf_dict))

    det_dict = {
                'random' : False,
                'homogeneous' : False,
                'deterministic': True,
                'confounded' : False,
                'seed' : 0
            }
    data_gen_list.append(ACICGenerator(det_dict))
    return data_gen_list

# Define Experiment
methods = [DoubleRobust(LinearRegression(), LinearRegression()),
           SingleOutcomeRegression(RandomForestRegressor()),
           PropensityScoreWeighting(LinearRegression()),
            DoubleOutcomeRegression(RandomForestRegressor())]

datasets = [IHDPDataProvider()]
metrics = [StandardEvaluation(ex)]
sizes = None


@ex.main
def run(_run):
    output = pd.DataFrame(columns=['metric', 'method', 'dataset', 'size', 'sample', 'time', 'score'])

    start = time.time()

    # Enforce right order of iteration
    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                metric.evaluate(dataset, method, sizes)
                output = output.append(metric.output, ignore_index=True)

    elapsed = time.time() - start
    print('++++++++++++++ ELAPSED TIME ++++++++++++++ ', elapsed, ' seconds')

    file_name = config.OUTPUT_PATH + str(_run._id) + '_output.csv'
    output.to_csv(file_name)
    _run.add_artifact(filename=file_name)


if __name__ == '__main__':
    ex.run()
