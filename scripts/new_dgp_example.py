import numpy as np
from sklearn.ensemble import RandomForestRegressor

from justcause.data.generators.example import example_emcs
from justcause.data.sets.ihdp import load_ihdp
from justcause.methods.basics.outcome_regression import SLearner

ihdp = load_ihdp()
replication = ihdp.data.loc[ihdp.data["rep"] == 3]  # re
# TODO: Rewrite fit function to just take a dataframe of the replication
x, t, y = (
    replication[ihdp.covariate_names],
    replication["t"],
    replication["y"],
)  # retrieve training data
learner = SLearner(RandomForestRegressor())
learner.fit(x, t, y)
ite = replication["ite"]
print(np.mean(ite))
print(np.mean(learner.predict_ite(x, t, y)))


example = example_emcs()
x, t, y = example.data[example.covariate_names], example.data["t"], example.data["y"]
learner.fit(x, t, y)
ite = example.data["ite"]
print(np.mean(ite))
print(np.mean(learner.predict_ite(x, t, y)))
