# Dynamically set the environment variable for R_HOME as
# found by running python -m rpy2.situation from the terminal
import os

import pandas as pd

os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

# To make it work on MacOS
import matplotlib
matplotlib.use("MacOSX")

import seaborn as sns
sns.set(style="darkgrid")

import matplotlib.pyplot as plt

# Sacred
from sacred import Experiment
from sacred.observers import MongoObserver

from sklearn.ensemble.forest import RandomForestRegressor, DecisionTreeRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from causaleval.data.data_provider import DataProvider
from causaleval.metrics import EvaluationMetric, StandardEvaluation
from causaleval.methods.causal_method import CausalMethod
from causaleval.methods.basics.outcome_regression import SingleOutcomeRegression, DoubleOutcomeRegression
from causaleval.methods.causal_forest import CausalForest
from causaleval.data.sets.ihdp import IHDPDataProvider
from causaleval.data.sets.twins import TwinsDataProvider
from causaleval import config

ex = Experiment('normal')
ex.observers.append(MongoObserver.create(url=config.DB_URL, db_name=config.DB_NAME))



@ex.main
def run_experiment(_run):

    output = pd.DataFrame(columns=['metric', 'method', 'dataset', 'score'])

    methods = [DoubleOutcomeRegression(DecisionTreeRegressor()), SingleOutcomeRegression(GradientBoostingRegressor())]
    datasets = [IHDPDataProvider(), TwinsDataProvider()]
    metrics = [StandardEvaluation(ex)]

    # Enfore right order of iteration
    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                metric.evaluate(dataset, method)
                output = output.append(metric.output, ignore_index=True)

    file_name = config.OUTPUT_PATH + str(_run._id) + '_output.csv'
    output.to_csv(file_name)
    _run.add_artifact(filename=file_name)


if __name__ == '__main__':
    ex.run()