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
    result = evaluate_ite(reps, learner, pehe_score, train_size=0.8)
    row = result[0]
    assert len(result) == 2
    assert len(row) == 5  # 2 standard + 3 formats for one metric
    assert "pehe_score-mean" in row.keys()  # three format per metric are reported


def test_summary():
    data = {"pehe_score": np.full(10, 1)}
    summary = summarize_scores(data)
    assert len(summary) == 3  # 5 pseudo-metrics times 3 formats
    assert summary["pehe_score-mean"] == 1

    # Also works with pd.DataFrame
    df = pd.DataFrame(data)
    summary = summarize_scores(df)
    assert len(summary) == 3  # 5 pseudo-metrics times 3 formats
    assert summary["pehe_score-mean"] == 1

    data = np.arange(10).reshape((-1, 1))
    df = pd.DataFrame(data)
    values = list(summarize_scores(df).values())
    assert values[0] == np.mean(data)


def test_calc_scores():
    true = np.full(100, 1)
    pred = np.full(100, 0)
    score_dict = calc_scores(true, pred, pehe_score)
    assert list(score_dict.values())[0] == 1
    assert "pehe_score" in score_dict.keys()
