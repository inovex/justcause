"""Contains default evaluation procedures and helpers for manual evaluation

The procedure used in `justcause.evaluation.evaluate_ite` is the recommended standard
for evaluating ITE estimation methods.

"""
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

from .data import CausalFrame, Col

#: Type aliases
Metric = Callable[[np.array, np.array], float]
Format = Callable[[Union[np.array, List[np.array]]], Union[float, List[float]]]
Frame = Union[CausalFrame, pd.DataFrame]

METHOD = "method"
TRAIN = "train"


def format_metric(metric, form):
    """Returns a string representation for metric-format combinations"""
    if callable(metric):
        metric_string = metric.__name__
    else:
        metric_string = str(metric)
    return "{}-{}".format(metric_string, form.__name__)


def evaluate_ite(
    replications: Union[CausalFrame, List[CausalFrame]],
    methods,
    metrics: Union[List[Metric], Metric],
    formats: Union[List[Format], Format] = (np.mean, np.median, np.std),
    train_size: float = 0.8,
    random_state: Optional[RandomState] = None,
) -> List[dict]:
    """Evaluate methods with multiple metrics on a given set of replications

    Good for use with standard causal methods and callables on new datasets.
    See Chapter "Usage" in the docs for an example.
    ITE prediction and evaluation is the most common setting,
    which is why this is automated, while other settings like ATE estimation are left
    to the user for now.

    Args:
        replications: One or more CausalFrames for each replication
        methods: Causal methods with `fit` and `predict_ite` methods
        metrics: metrics to score the ITE predictions
        formats: formats to summarize metrics over multiple replications
        train_size: ratio of training data in each replication
        random_state: random_state passed to train_test_split

    Returns: A DataFrame with the results in a structured manner
        One for each (method, train/test) pair

    """
    if not isinstance(methods, list):
        methods = [methods]

    if not isinstance(metrics, list):
        metrics = [metrics]

    if not isinstance(replications, list):
        replications = [replications]

    results = list()

    for method in methods:

        train_result, test_result = _evaluate_single_method(
            replications, method, metrics, formats, train_size, random_state
        )

        if callable(method):
            name = method.__name__
        else:
            name = str(method)

        # Add run
        train_result.update({METHOD: name, TRAIN: True})
        test_result.update({METHOD: name, TRAIN: False})

        results.append(train_result)
        results.append(test_result)

    return results


def _evaluate_single_method(
    replications,
    method,
    metrics,
    formats=(np.mean, np.median, np.std),
    train_size=0.8,
    random_state=None,
) -> Tuple[dict, dict]:
    """Helper to evaluate method with multiple metrics on the given replications.

    This is the standard variant of an evaluation loop, which the user can implement
    manually to modify parts of it. Here, only ITE prediction and evaluation is
    considered.

    Returns:
        a tuple of two dicts which map (score_name) -> (score)
        summarized over all replications for train and test respectively

    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    if not isinstance(replications, list):
        replications = [replications]

    train_scores = list()
    test_scores = list()

    for rep in replications:
        train, test = train_test_split(
            rep, train_size=train_size, random_state=random_state
        )

        if callable(method):
            train_ite, test_ite = method(train, test)
        else:
            train_ite, test_ite = default_predictions(method, train, test)

        train_scores.append(calc_scores(train[Col.ite], train_ite, metrics))
        test_scores.append(calc_scores(test[Col.ite], test_ite, metrics))

    train_results = summarize_scores(train_scores, formats)
    test_results = summarize_scores(test_scores, formats)

    return train_results, test_results


def calc_scores(
    true: np.array, pred: np.array, metrics: Union[List[Metric], Metric]
) -> dict:
    """Compare ground-truth to predictions with given metrics for one replication

    Call for train and test separately.

    Args:
        true: true ITE
        pred: predicted ITE
        metrics: metrics to evaluate on the ITEs

    Returns:
        dict: a dict of (score_name, scores) pairs with len(metrics) entries

    """
    # ensure `metrics` is a list for use in list comprehension
    if not isinstance(metrics, list):
        metrics = [metrics]

    return {metric.__name__: metric(true, pred) for metric in metrics}


def default_predictions(
    method, train: CausalFrame, test: CausalFrame
) -> Tuple[np.array, np.array]:
    """Returns the default predictions of the causal method after training on train

    Convenience method to use with standard methods.

    Args:
        method: Causal method with `fit` and `predict_ite` methods
        train: the train CausalFrame
        test: the test CausalFrame

    Returns: (train_ite, test_ite), ITE predictions for train and test

    """
    # get numpy data out of CausalFrame
    train_X, train_t, train_y = train.np.X, train.np.t, train.np.y
    test_X, test_t, test_y = test.np.X, test.np.t, test.np.y

    method.fit(train_X, train_t, train_y)

    train_ite = method.predict_ite(train_X, train_t, train_y)
    test_ite = method.predict_ite(test_X, test_t, test_y)

    return train_ite, test_ite


def summarize_scores(
    scores: Union[pd.DataFrame, List[dict]],
    formats: Union[List[Format], Format] = (np.mean, np.median, np.std),
) -> np.array:
    """

    Call for train and test separately

    Args:
        scores: the DataFrame or DataFrame-like containing scores for all replications
        formats: Summaries to calculate over the scores of multiple replications

    Returns:
        dict: a dictionary mapping

    """
    # make sure we're dealing with pd.DataFrame
    df = pd.DataFrame(scores)
    dict_of_results = {
        format_metric(metric, form): form(df[metric])
        for metric in df.columns
        for form in formats
    }
    return dict_of_results
