import pytest

# TODO: Fix this workaround here and make config a yaml in the configs folder.
#       We exploit here the fact that Python only imports once and then
#       keeps a lookup of it.
import config  # noqa: F401
from sklearn.linear_model import LinearRegression

from justcause.data.generators.toy import SWagerDataProvider
from justcause.methods.basics.outcome_regression import SLearner
from justcause.methods.causal_forest import CausalForest
from justcause.metrics import StandardEvaluation


def test_experiment():
    """
    Tests the integration of method, data and metric into an experiment

    requires a directory `results` for the FileStorageObserver

    :return:
    """
    method = SLearner(LinearRegression())
    data = SWagerDataProvider()
    metric = StandardEvaluation(sizes=None, num_runs=1)
    metric.evaluate(data, method)
    assert len(metric.output.index) == 8  # 4 scores on train/test each
    assert metric.output["score"][0] != 0

    # test multirun
    metric = StandardEvaluation(sizes=None, num_runs=5)
    metric.evaluate(data, method)

    # test varying sizes
    metric = StandardEvaluation(sizes=[100, 200], num_runs=5)
    metric.evaluate(data, method)
    assert len(metric.output.index) == 16


def test_dataprovider():
    data = SWagerDataProvider()
    x, t, y = data.get_training_data(size=500)
    assert len(t) == len(y)
    assert len(y) == len(x)
    assert len(t) == 500

    with pytest.raises(AssertionError):
        data.get_training_data(size=2001)

    data.set_train_test_split(train_size=0.5)
    x, t, y = data.get_training_data(size=1000)
    assert len(t) == 0.5 * len(data.t)
    x_test, t_test, y_test = data.get_test_data()
    assert len(t_test) == 1000


def test_rpy2(grf):
    """
    Tests whether rpy2 is able to load the R environment and
    execute a causal forest
    """
    data = SWagerDataProvider()
    cf = CausalForest()
    cf.fit(*data.get_training_data())
    assert cf.predict_ite(*data.get_test_data()) is not None
