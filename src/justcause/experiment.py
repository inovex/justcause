# Dynamically set the environment variable for R_HOME as
# found by running python -m rpy2.situation from the terminal
import os
import time
import pandas as pd

import config

os.environ['L_ALL'] = 'en_US.UTF-8'
os.environ['R_HOME'] = config.R_HOME

# Sacred
from sacred import Experiment
from sacred.observers import MongoObserver

# Base Methods
from sklearn.ensemble.forest import RandomForestRegressor, DecisionTreeRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

from .metrics import EvaluationMetric, StandardEvaluation

# Methods
from src.justcause.methods import CausalMethod
from src.justcause.methods import SingleOutcomeRegression, DoubleOutcomeRegression
from src.justcause.methods import DoubleRobust
from src.justcause.methods import Bart
from src.justcause.methods import PropensityScoreWeighting
from src.justcause.methods import CausalForest
from src.justcause.methods import GANITEWrapper
from src.justcause.methods import DragonNetWrapper

# Data
from causaleval.data.generators.acic import ACICGenerator
from causaleval.data.generators.ihdp import IHDPGenerator
from causaleval.data.generators.toy import SWagerRealCompare, SWagerDataProvider, CovariateModulator
from causaleval.data.sets.ihdp import IHDPDataProvider, IHDPCfrProvider
from causaleval.data.sets.twins import TwinsDataProvider
from causaleval.data.sets.ibm import SimpleIBMDataProvider

ex = Experiment('normal')
ex.observers.append(MongoObserver.create(url=config.DB_URL, db_name=config.DB_NAME))

def create_data_gens():
    # tf_list = [True, False]
    # data_gen_list = []
    #
    # for one in tf_list:
    #     for two in tf_list:
    #         dict = {
    #             'random' : one,
    #             'homogeneous' : two,
    #             'deterministic': False,
    #             'confounded' : False,
    #             'seed' : 0
    #         }
    #         data_gen_list.append(ACICGenerator(dict))
    pass

homo_dict =  {
            'random' : False,
            'homogeneous' : True,
            'deterministic': False,
            'confounded' : False,
            'seed' : 0
        }

conf_dict = {
            'random' : False,
            'homogeneous' : False,
            'deterministic': False,
            'confounded' : True,
            'seed' : 0
        }
# data_gen_list.append(ACICGenerator(conf_dict))

det_dict = {
            'random' : False,
            'homogeneous' : False,
            'deterministic': True,
            'confounded' : False,
            'seed' : 0
        }

datasets = [IHDPCfrProvider()]

# Define Experiment
methods = [
    DoubleOutcomeRegression(LinearRegression(), LinearRegression()),
           ]

sizes = None
metrics = [StandardEvaluation(ex, sizes=sizes, num_runs=1000)]


@ex.automain
def run(_run):

    print('start')
    output = pd.DataFrame(columns=['metric', 'method', 'dataset', 'size', 'sample', 'time', 'score', 'std'])

    start = time.time()
    if datasets and methods and sizes:
        num_experiments = len(datasets)*len(methods)*len(sizes)
        print('running {} training runs'.format(num_experiments))


    # Enforce right order of iteration
    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                metric.evaluate(dataset, method, plot=False)
                output = output.append(metric.output, ignore_index=True)

    elapsed = time.time() - start
    print('++++++++++++++ ELAPSED TIME ++++++++++++++ ', elapsed, ' seconds')

    file_name = config.OUTPUT_PATH + str(_run._id) + '_output.csv'
    output.to_csv(file_name)
    _run.add_artifact(filename=file_name)


if __name__ == '__main__':
    # ex.run()
    pass
