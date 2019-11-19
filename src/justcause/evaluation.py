import itertools
from typing import Callable, Iterator, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

from .data import CausalFrame, Col

#: Type aliases

METRIC = Callable[[np.array, np.array], float]

Frame = Union[CausalFrame, pd.DataFrame]

STD_COL = ["method", "train", "num_rep"]
SCORE_FORMATS = ["mean", "median", "std"]


def make_result_df(metrics: List[METRIC]):
    cols = STD_COL + [
        "{0}-{1}".format(metric.__name__, form)
        for metric in metrics
        for form in SCORE_FORMATS
    ]
    return pd.DataFrame(columns=cols)


def evaluate(
    replications: Union[CausalFrame, List[CausalFrame]],
    methods,
    metrics: Union[List[METRIC], METRIC],
    train_size: int = 0.8,
    random_state: Optional[RandomState] = None,
) -> pd.DataFrame:
    """
    Evaluate multiple methods with multiple metrics on a given set of replications

    Args:
        replications:
        methods:
        metrics:
        train_size:
        random_state:

    Returns:

    """
    if not isinstance(methods, list):
        methods = [methods]

    if not isinstance(metrics, list):
        metrics = [metrics]

    if not isinstance(replications, list):
        replications = [replications]

    results_df = make_result_df(metrics)

    for method in methods:

        multi_result = _evaluate_single_method(
            replications, method, metrics, train_size, random_state
        )
        results_df = _append_evaluation_rows(
            results_df, multi_result, str(method), len(replications)
        )

    return results_df


def evaluate_all(
    datasets: List[Iterator],
    data_names: List[str],
    methods,
    metrics: Union[List[METRIC], METRIC],
    num_replications: int,
    train_size: int = 0.8,
    random_state: Optional[RandomState] = None,
) -> pd.DataFrame:
    """
    Evaluates multiple methods on multiple datasets with given metrics

    Args:
        datasets: a list of iterators providing the replications to be evaluated on
        data_names: a list of data names used in the dataframe to identify datasets
        methods: a list of methods which to evaluate
        metrics: a list of metrics used to score the methods
        num_replications: number of replications for each dataset
        train_size: ratio of data used for training per replication
        random_state: random state passed to train_test_split

    Returns:

    """

    results_df = make_result_df(metrics)
    results_df["data"] = "placeholder"

    for i, data_it in enumerate(datasets):

        # TODO: Check if enough replications are available? Do we have
        #  to load all in order to check for that?
        replications = list(itertools.islice(data_it, num_replications))

        data_res = evaluate(replications, methods, metrics, train_size, random_state)
        data_res["data"] = data_names[i]
        results_df = results_df.append(data_res, ignore_index=True)

    return results_df


def get_train_test_predictions(
    method, train: CausalFrame, test: CausalFrame, out_of_sample: bool = False
):
    """
    Returns the predictions of the given method after training on train

    Args:
        method: Causal method with `fit` and `predict_ite` methods
        train: the train CausalFrame
        test: the test CausalFrame
        out_of_sample: whether or not out-of-sample predictions should be returned, too

    Returns:

    """
    # get numpy data out of CausalFrame
    train_X, train_t, train_y = train.np.X, train.np.t, train.np.y
    test_X, test_t, test_y = test.np.X, test.np.t, test.np.y

    method.fit(train_X, train_t, train_y)

    if out_of_sample:
        train_ite = method.predict_ite(train_X)
        test_ite = method.predict_ite(test_X)
    else:
        train_ite = method.predict_ite(train_X, train_t, train_y)
        test_ite = method.predict_ite(test_X, test_t, test_y)

    return train_ite, test_ite


def get_train_test_scores(
    replications: Union[CausalFrame, List[CausalFrame]],
    method,
    metrics: Union[METRIC, List[METRIC]],
    train_size: int = 0.8,
    random_state: Optional[RandomState] = None,
):
    """
    Compute the scores of a metric for all replications separately

    The scores computed here are used to summarize the performance across
    multiple replications as mean, median, ...

    Args:
        replications: replications on which to evaluate
        method: method to evaluate
        metric: list of one or more metrics used for evaluation
        train_size: ratio of training data used
        random_state: random_state passed to train_test_split

    Returns: The scores of a metric on each replication used to calculate

    """
    # ensure metrics and replications are lists, even if with just one element
    if not isinstance(metrics, list):
        metrics = [metrics]

    if not isinstance(replications, list):
        replications = [replications]

    train_scores = np.zeros((len(replications), len(metrics)))
    test_scores = np.zeros((len(replications), len(metrics)))

    for i, rep in enumerate(replications):
        train, test = train_test_split(
            rep, train_size=train_size, random_state=random_state
        )

        train_ite, test_ite = get_train_test_predictions(method, train, test)
        train_scores[i, :] = np.array(
            [metric(train[Col.ite], train_ite) for metric in metrics]
        )
        test_scores[i, :] = np.array(
            [metric(test[Col.ite], test_ite) for metric in metrics]
        )

    return train_scores, test_scores


def _evaluate_single_method(
    replications, method, metrics, train_size=0.8, random_state=None
):
    """ Helper to evaluate a method with multiple metrics on the given replications"""

    train_scores, test_scores = get_train_test_scores(
        replications, method, metrics, train_size, random_state
    )

    train_results = [
        np.mean(train_scores, axis=0),
        np.median(train_scores, axis=0),
        np.std(train_scores, axis=0),
    ]
    test_results = [
        np.mean(test_scores, axis=0),
        np.median(test_scores, axis=0),
        np.std(test_scores, axis=0),
    ]

    # Returns the mean, median, std. for all metrics over all replications
    return train_results, test_results


def _make_row(metric_tuple, method_name, train, num_rep):
    """ Format helper to make a row out of the score tuples"""
    mean, median, std = metric_tuple
    scores = [x for t in zip(mean, median, std) for x in t]
    row = [method_name, train, num_rep] + scores
    return row


def _append_evaluation_rows(df, multi_metric_result, method_name, num_rep):
    """ Reformat results into a dataframe row"""
    train, test = multi_metric_result
    train_row = _make_row(train, method_name, 1, num_rep)
    df.loc[len(df)] = train_row
    test_row = _make_row(test, method_name, 0, num_rep)
    df.loc[len(df)] = test_row
    return df
