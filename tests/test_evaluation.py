from itertools import islice

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from justcause.evaluation import evaluate, evaluate_all
from justcause.learners import SLearner
from justcause.metrics import mean_absolute, pehe_score


def test_single_evaluation(ihdp_data):
    reps = list(islice(ihdp_data, 10))
    learner = SLearner(LinearRegression())
    df = evaluate(reps, learner, pehe_score, 0.8)
    assert len(df) == 2  # three format per metric are reported
    assert "pehe_score-mean" in df.columns  # three format per metric are reported


def test_evaluate_all(ihdp_data, twins_data):
    datasets = [ihdp_data, twins_data]
    data_names = ["ihdp", "twins"]
    learners = [SLearner(LinearRegression()), SLearner(RandomForestRegressor())]
    metrics = [pehe_score, mean_absolute]

    df = evaluate_all(datasets, data_names, learners, metrics, 1)
    assert len(df) == 8  # 8 rows. one for each (data, learner, train/test)

    assert "pehe_score-mean" in df.columns  # three format per metric are reported
    assert "mean_absolute-mean" in df.columns  # three format per metric are reported
    assert "mean_absolute-median" in df.columns  # three format per metric are reported
    assert "mean_absolute-std" in df.columns  # three format per metric are reported
