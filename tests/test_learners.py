import pytest

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from justcause.learners import (
    RLearner,
    SLearner,
    TLearner,
    WeightedSLearner,
    WeightedTLearner,
)
from justcause.learners.utils import replace_factual_outcomes


def test_slearner(ihdp_data):
    rep = next(ihdp_data)
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    slearner = SLearner(LinearRegression())
    slearner.fit(x, t, y)
    pred = slearner.predict_ite(x, t, y)
    assert len(pred) == len(t)
    assert str(slearner) == "SLearner(regressor=LinearRegression)"


def test_weighted_slearner(ihdp_data):

    with pytest.raises(AssertionError):
        # Raises a 'missing predict_proba' error
        WeightedSLearner(LinearRegression(), LinearRegression())

    rep = next(ihdp_data)
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    slearner = WeightedSLearner(LinearRegression())
    slearner.fit(x, t, y)
    pred = slearner.predict_ite(x, t, y)
    assert len(pred) == len(t)
    assert (
        str(slearner) == "WeightedSLearner(regressor=LinearRegression, "
        "propensity=ElasticNetPropensityModel)"
    )

    pred, y_0, y_1 = slearner.predict_ite(
        x, t, y, return_components=True, replace_factuals=True
    )
    assert y_0 is not None
    assert y_1 is not None
    union = np.c_[y_0, y_1]
    assert union[0, int(t[0])] == y[0]  # factuals were replaced

    slearner = WeightedSLearner(LinearRegression(), LogisticRegression())
    slearner.fit(x, t, y)
    pred = slearner.predict_ite(x, t, y)
    assert len(pred) == len(t)

    ate = slearner.predict_ate(x, t, y)
    assert abs(ate - 4.0) < 0.1

    pred, y_0, y_1 = slearner.predict_ite(
        x, t, y, return_components=True, replace_factuals=False
    )
    assert np.all(pred == y_1 - y_0)


def test_tlearner(ihdp_data):
    """Construct T-Learner in different ways and ensure it works"""
    rep = next(ihdp_data)
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    tlearner = TLearner(LinearRegression())
    tlearner.fit(x, t, y)
    pred = tlearner.predict_ite(x, t, y)
    assert len(pred) == len(t)
    assert (
        str(tlearner) == "TLearner(control=LinearRegression, "
        "treated=LinearRegression)"
    )

    ate = tlearner.predict_ate(x, t, y)
    assert abs(ate - 4.0) < 0.1

    tlearner = TLearner(learner_c=LinearRegression(), learner_t=RandomForestRegressor())
    tlearner.fit(x, t, y)
    pred = tlearner.predict_ite(x, t, y)

    assert len(pred) == len(t)
    assert (
        str(tlearner) == "TLearner(control=LinearRegression, "
        "treated=LinearRegression)"
    )

    ate = tlearner.predict_ate(x, t, y)
    assert abs(ate - 4.0) < 0.1

    tlearner = TLearner(LinearRegression())
    tlearner.fit(x, t, y)
    pred = tlearner.predict_ite(x, t, y)
    assert len(pred) == len(t)


def test_weighted_tlearner(ihdp_data):
    rep = next(ihdp_data)
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    tlearner = WeightedTLearner(
        learner=LinearRegression(), propensity_learner=LogisticRegression()
    )
    tlearner.fit(x, t, y)
    pred = tlearner.predict_ite(x, t, y)
    assert len(pred) == len(t)

    tlearner = WeightedTLearner(learner=LinearRegression())
    assert (
        str(tlearner) == "WeightedTLearner(control=LinearRegression, "
        "treated=LinearRegression, "
        "propensity=ElasticNetPropensityModel)"
    )

    prop = np.full(len(t), 1)
    tlearner.fit(x, t, y, propensity=prop)
    pred = tlearner.predict_ite(x, t, y)
    assert len(pred) == len(t)


def test_rlearner(ihdp_data):
    rep = next(ihdp_data)
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    rlearner = RLearner(LinearRegression())
    rlearner.fit(x, t, y)
    pred = rlearner.predict_ite(x)
    assert len(pred) == len(t)
    assert (
        str(rlearner) == "RLearner(outcome=LinearRegression, effect=LinearRegression)"
    )
