# TODO: Completely rework or better delete, use an example in the notebook instead
# Dynamically set the environment variable for R_HOME as
# found by running python -m rpy2.situation from the terminal
import time

import config
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.linear_model import LinearRegression

from justcause.data.sets.ihdp import IHDPCfrProvider
from justcause.learners.meta.outcome_regression import DoubleOutcomeRegression
from justcause.metrics import StandardEvaluation

DB_NAME = "sacred"
DB_URL = "localhost:27017"

ex = Experiment("normal")
ex.observers.append(MongoObserver.create(url=DB_URL, db_name=DB_NAME))


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


homo_dict = {
    "random": False,
    "homogeneous": True,
    "deterministic": False,
    "confounded": False,
    "seed": 0,
}

conf_dict = {
    "random": False,
    "homogeneous": False,
    "deterministic": False,
    "confounded": True,
    "seed": 0,
}
# data_gen_list.append(ACICGenerator(conf_dict))

det_dict = {
    "random": False,
    "homogeneous": False,
    "deterministic": True,
    "confounded": False,
    "seed": 0,
}

datasets = [IHDPCfrProvider()]

# Define Experiment
methods = [DoubleOutcomeRegression(LinearRegression(), LinearRegression())]

sizes = None
metrics = [StandardEvaluation(ex, sizes=sizes, num_runs=1000)]


@ex.automain
def run(_run):

    print("start")
    output = pd.DataFrame(
        columns=[
            "metric",
            "method",
            "dataset",
            "size",
            "sample",
            "time",
            "score",
            "std",
        ]
    )

    start = time.time()
    if datasets and methods and sizes:
        num_experiments = len(datasets) * len(methods) * len(sizes)
        print("running {} training runs".format(num_experiments))

    # Enforce right order of iteration
    for dataset in datasets:
        for method in methods:
            for metric in metrics:
                metric.evaluate(dataset, method, plot=False)
                output = output.append(metric.output, ignore_index=True)

    elapsed = time.time() - start
    print("++++++++++++++ ELAPSED TIME ++++++++++++++ ", elapsed, " seconds")

    file_name = config.OUTPUT_PATH + str(_run._id) + "_output.csv"
    output.to_csv(file_name)
    _run.add_artifact(filename=file_name)


if __name__ == "__main__":
    # ex.run()
    pass
