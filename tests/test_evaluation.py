from itertools import islice

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from justcause.evaluation import (
    calc_scores,
    evaluate_ite,
    setup_result_df,
    setup_scores_df,
    summarize_scores,
)
from justcause.learners import SLearner
from justcause.metrics import enormse, pehe_score


def test_single_evaluation(ihdp_data):
    reps = list(islice(ihdp_data, 10))
    learner = SLearner(LinearRegression())
    df = evaluate_ite(reps, learner, pehe_score, train_size=0.8)
    assert len(df) == 2
    assert len(df.columns) == 5  # 2 standard + 3 formats for one metric
    assert "pehe_score-mean" in df.columns  # three format per metric are reported


def test_summary():
    metrics = [pehe_score]
    df = setup_scores_df(metrics)
    df["pehe_score"] = np.full(10, 1)
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


def test_setup_df():

    metrics = [pehe_score]

    df = setup_scores_df(metrics)
    assert len(df.columns) == 1
    assert "pehe_score" in df.columns

    metrics = [pehe_score, enormse]
    df = setup_scores_df(metrics)
    assert len(df.columns) == 2
    assert "enormse" in df.columns

    result = setup_result_df(metrics)
    assert len(result.columns) == 8  # 2 base + 3 for each metric
    assert "pehe_score-mean" in result.columns

    formats = [np.mean, np.std]
    result = setup_result_df(metrics, formats)
    assert len(result.columns) == 6  # 2 base + 2 for each metric
    assert "enormse-std" in result.columns
