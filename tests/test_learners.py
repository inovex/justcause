import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from justcause.learners import (
    CausalForest,
    DoubleRobustEstimator,
    DragonNet,
    PSWEstimator,
    RLearner,
    SLearner,
    TLearner,
    XLearner,
)
from justcause.metrics import pehe_score


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


def test_rlearner(ihdp_data):
    rep = ihdp_data[0]
    x, t, y = rep.np.X, rep.np.t, rep.np.y

    # with RF explicitly
    rlearner = RLearner(RandomForestRegressor())
    rlearner.fit(x, t, y)
    pred = rlearner.predict_ite(x)
    assert len(pred) == len(t)

    # With default
    rlearner = RLearner()
    rlearner.fit(x, t, y)
    pred = rlearner.predict_ite(x)
    assert len(pred) == len(t)

    assert (
        str(rlearner) == "RLearner(outcome=LinearRegression, effect=LinearRegression)"
    )


def test_xlearner(ihdp_data):
    rep = ihdp_data[0]
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    true_ite = rep["ite"].values

    # With LinearRegression
    xlearner = XLearner(LinearRegression())
    xlearner.fit(x, t, y)
    pred = xlearner.predict_ite(x, t, y)
    assert len(pred) == len(t)

    # With default
    xlearner = XLearner(LinearRegression())
    xlearner.fit(x, t, y)
    pred = xlearner.predict_ite(x, t, y)
    assert len(pred) == len(t)
    assert abs(pehe_score(true_ite, pred) - 0.5) < 0.2

    pred_ate = xlearner.estimate_ate(x, t, y)
    true_ate = np.mean(rep["ite"].values)
    assert abs(pred_ate - true_ate) < 0.2


def test_causalforest(ihdp_data, grf):
    rep = ihdp_data[0]
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    cf = CausalForest()
    cf.fit(x, t, y)
    pred_ate = cf.estimate_ate(x, t, y)
    true_ate = np.mean(rep["ite"].values)
    assert abs(pred_ate - true_ate) < 0.2

    # Try passing keyword arguments to the R implenetation
    cf = CausalForest(num_trees=50, alpha=0.1, honesty=False)
    cf.fit(x, t, y)
    pred_ate = cf.estimate_ate(x, t, y)
    true_ate = np.mean(rep["ite"].values)
    assert abs(pred_ate - true_ate) < 0.2


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


def test_dragonnet(ihdp_data):

    rep = ihdp_data[0]
    x, t, y = rep.np.X, rep.np.t, rep.np.y
    dragon = DragonNet()
    dragon.fit(x, t, y)
    ate = np.mean(dragon.predict_ite(x, t, y))

    true_ate = np.mean(rep["ite"].values)
    assert abs(ate - true_ate) < 0.5
