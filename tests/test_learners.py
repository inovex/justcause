import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from justcause.learners import DoubleRobustEstimator, PSWEstimator, SLearner, TLearner


def test_slearner(ihdp_data):
    rep = ihdp_data[0]
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    slearner = SLearner(LinearRegression())
    slearner.fit(x, t, y)
    pred = slearner.predict_ite(x, t, y)
    assert len(pred) == len(t)
    assert str(slearner) == "SLearner(learner=LinearRegression)"

    true_ate = np.mean(rep["ite"].values)

    ate = slearner.estimate_ate(x, t, y)
    assert abs(ate - true_ate) < 0.2


def test_tlearner(ihdp_data):
    """Construct T-Learner in different ways and ensure it works"""
    rep = ihdp_data[0]
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    tlearner = TLearner(LinearRegression())
    tlearner.fit(x, t, y)
    pred = tlearner.predict_ite(x, t, y)
    assert len(pred) == len(t)
    assert (
        str(tlearner) == "TLearner(control=LinearRegression, "
        "treated=LinearRegression)"
    )

    true_ate = np.mean(rep["ite"].values)

    ate = tlearner.estimate_ate(x, t, y)
    assert abs(ate - true_ate) < 0.2

    tlearner = TLearner(
        learner_c=LinearRegression(), learner_t=RandomForestRegressor(random_state=42)
    )
    tlearner.fit(x, t, y)
    pred = tlearner.predict_ite(x, t, y)

    assert len(pred) == len(t)
    assert (
        str(tlearner) == "TLearner(control=LinearRegression, "
        "treated=RandomForestRegressor)"
    )

    ate = tlearner.estimate_ate(x, t, y)
    assert abs(ate - true_ate) < 0.2

    tlearner = TLearner(LinearRegression())
    tlearner.fit(x, t, y)
    pred = tlearner.predict_ite(x, t, y)
    assert len(pred) == len(t)


def test_dre(ihdp_data):
    rep = ihdp_data[0]
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    dre = DoubleRobustEstimator(LogisticRegression(random_state=42))
    ate = dre.estimate_ate(x, t, y)
    true_ate = np.mean(rep["ite"].values)
    assert abs(ate - true_ate) < 0.3

    # With default learner
    ate = dre.estimate_ate(x, t, y)
    assert abs(ate - true_ate) < 0.4


def test_psw(ihdp_data):
    rep = ihdp_data[0]
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    psw = PSWEstimator(LogisticRegression(random_state=42))
    ate = psw.estimate_ate(x, t, y)
    true_ate = np.mean(rep["ite"].values)
    assert abs(ate - true_ate) < 1

    psw = PSWEstimator()
    ate = psw.estimate_ate(x, t, y)
    assert ate > 0
