from itertools import islice

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from justcause.evaluation import calc_scores, evaluate_ite, summarize_scores
from justcause.learners import SLearner
from justcause.metrics import pehe_score


def test_single_evaluation(ihdp_data):
    reps = list(islice(ihdp_data, 10))
    learner = SLearner(LinearRegression())
    df = evaluate_ite(reps, learner, pehe_score, train_size=0.8)
    assert len(df) == 2
    assert len(df.columns) == 5  # 2 standard + 3 formats for one metric
    assert "pehe_score-mean" in df.columns  # three format per metric are reported


def test_summary():
    data = np.full((10, 5), 1)
    df = pd.DataFrame(data)
    summary = summarize_scores(df)
    assert len(summary) == 15  # 5 pseudo-metrics times 3 formats
    assert summary[0] == 1

    data = np.arange(10).reshape((-1, 1))
    df = pd.DataFrame(data)
    assert summarize_scores(df)[0] == np.mean(data)


def test_calc_scores():
    true = np.full(100, 1)
    pred = np.full(100, 0)
    assert calc_scores(true, pred, pehe_score)[0] == 1
