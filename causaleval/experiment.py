# Dynamically set the environment variable for R_HOME as
# found by running python -m rpy2.situation from the terminal
import os
import time
import pandas as pd
from causaleval import config

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
from sklearn.linear_model import LogisticRegression


from causaleval.metrics import EvaluationMetric, StandardEvaluation

# Methods
from causaleval.methods.causal_method import CausalMethod
from causaleval.methods.causal_learning import CausalBoosting, PolinatedTransformedOutcomeForest, CausalMars
from causaleval.methods.r_learner import RLearner, XLearner
from causaleval.methods.basics.outcome_regression import SingleOutcomeRegression, DoubleOutcomeRegression
from causaleval.methods.basics.double_robust import DoubleRobust
from causaleval.methods.basics.bart import Bart
from causaleval.methods.basics.propensity_weighting import PropensityScoreWeighting
from causaleval.methods.causal_forest import CausalForest
from causaleval.methods.ganite_wrapper import GANITEWrapper
from causaleval.methods.dragonnet_wrapper import DragonNetWrapper

# Data
from causaleval.data.generators.acic import ACICGenerator
from causaleval.data.generators.ihdp import IHDPGenerator
from causaleval.data.generators.toy import SWagerRealCompare, SWagerDataProvider, CovariateModulator
from causaleval.data.sets.ihdp import IHDPDataProvider, IHDPReplicaProvider, IHDPCfrProvider
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
