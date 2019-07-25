# Dynamically set the environment variable for R_HOME as
# found by running python -m rpy2.situation from the terminal
import os
os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'

from causaleval.methods.basics.outcome_regression import SingleOutcomeRegression
from causaleval.methods.causal_forest import CausalForest
from causaleval.data.sets.ihdp import IHDPDataProvider
from sklearn.ensemble.forest import RandomForestRegressor

# Testwise implementation


provider = IHDPDataProvider()

cf = CausalForest(0)
cf.fit(*provider.get_training_data())
method = SingleOutcomeRegression(0, RandomForestRegressor())

method.fit(*provider.get_training_data())
x, _, _ = provider.get_training_data()
print(method.predict_ite(x))
print(cf.predict_ite(x))

