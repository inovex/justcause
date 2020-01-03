from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split

from .data import CausalFrame, Col

#: Type aliases
Metric = Callable[[np.array, np.array], float]
Format = Callable[[Union[np.array, List[np.array]]], Union[float, List[float]]]
Frame = Union[CausalFrame, pd.DataFrame]

STD_COL = ["method", "train"]


def format_metric(metric, form):
    if callable(metric):
        metric_string = metric.__name__
    else:
        metric_string = str(metric)
    return "{}-{}".format(metric_string, form.__name__)


def setup_scores_df(metrics: Union[List[Metric], Metric]):
    """Setup DataFrame containing the metric scores for all replications

    Args:
        metrics: metrics used for naming the columns

    Returns: DataFrame to store the scores for each replication
    """
    cols = [metric.__name__ for metric in metrics]
    return pd.DataFrame(columns=cols)


def setup_result_df(
    metrics: Union[List[Metric], Metric], formats=(np.mean, np.median, np.std)
):
    """Setup DataFrame containing the summarized scores for all methods and datasets

    Args:
        metrics: metrics used for scoring
        formats: formats for summarizing metrics (e.g. mean, std, ...)

    Returns: DataFrame to store the results for each method
    """
    cols = STD_COL + [
        format_metric(metric, form) for metric in metrics for form in formats
    ]
    return pd.DataFrame(columns=cols)


def evaluate_ite(
    generator: Iterator[Frame],
    num_reps: int,
    methods,
    metrics: Union[List[Metric], Metric],
    formats: Union[List[Format], Format] = (np.mean, np.median, np.std),
    train_size: float = 0.8,
    random_state: Optional[RandomState] = None,
) -> pd.DataFrame:
    """Evaluate methods with multiple metrics on a given set of replications

    Good for use with standard causal methods and callables on new datasets.
    See `notebooks/example_evalutation.ipynb` for an example.
    ITE prediction and evaluation is the most common setting,
    which is why this is automated, while other settings like ATE estimation are left
    to the user for now.

    Args:
        generator: generator yielding the replications of a dataset
        num_reps: the number of replications used for evaluation
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

    results_df = setup_result_df(metrics, formats)

    for method in methods:

        train_result, test_result = _evaluate_single_method(
            generator, num_reps, method, metrics, formats, train_size, random_state
        )

        if callable(method):
            name = method.__name__
        else:
            name = str(method)

        train_result["method"] = name
        train_result["train"] = True

        test_result["method"] = name
        test_result["train"] = False

        results_df = results_df.append(train_result, ignore_index=True)
        results_df = results_df.append(test_result, ignore_index=True)

    return results_df


def _evaluate_single_method(
    generator: Iterator[Frame],
    num_reps: int,
    method,
    metrics: Union[List[Metric], Metric],
    formats: Union[List[Format], Format] = (np.mean, np.median, np.std),
    train_size: int = 0.8,
    random_state: RandomState = None,
):
    """Helper to evaluate method with multiple metrics on the given replications

    This is the standard variant of an evaluation loop, which the user can implement
    manually to modify parts of it. Here, only ITE prediction and evaluation is
    considered.

    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    test_scores = setup_scores_df(metrics)
    train_scores = setup_scores_df(metrics)

    for _ in range(num_reps):
        # Only instantiate replication data when needed
        rep = next(generator)
        train, test = train_test_split(
            rep, train_size=train_size, random_state=random_state
        )

        if callable(method):
            train_ite, test_ite = method(train, test)
        else:
            train_ite, test_ite = default_predictions(method, train, test)

        test_scores.loc[len(test_scores)] = calc_scores(
            test[Col.ite], test_ite, metrics
        )

        train_scores.loc[len(train_scores)] = calc_scores(
            train[Col.ite], train_ite, metrics
        )

    train_results = summarize_scores(train_scores, formats)
    test_results = summarize_scores(train_scores, formats)

    return train_results, test_results


def calc_scores(true: np.array, pred: np.array, metrics):
    """Compare ground-truth to predictions with given metrics for one replication

    Call for train and test separately

    Args:
        true: True ITE
        pred: predicted ITE
        metrics: metrics to evaluate on the ITEs

    Returns: a list of scores with length == len(metrics), i.e. the row to be added to
        the scores dataframe

    """
    # ensure metrics and replications are lists, even if with just one element
    if not isinstance(metrics, list):
        metrics = [metrics]

    return np.array([metric(true, pred) for metric in metrics])


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


def get_default_callable(method):
    """Helper to get an evaluation callable for standard methods

    Args:
        method: Method to use for the standard callable

    Returns: Callable for evaluation in custom loop
    """

    def default_callable(train, test):
        train_X, train_t, train_y = train.np.X, train.np.t, train.np.y
        test_X, test_t, test_y = test.np.X, test.np.t, test.np.y

        method.fit(train_X, train_t, train_y)

        train_ite = method.predict_ite(train_X, train_t, train_y)
        test_ite = method.predict_ite(test_X, test_t, test_y)

        return train_ite, test_ite

    return default_callable


def summarize_scores(
    scores_df: pd.DataFrame,
    formats: Union[List[Format], Format] = (np.mean, np.median, np.std),
) -> np.array:
    """

    Call for train and test separately

    Args:
        scores_df: the dataframe containing scores for all replications
        formats: Summaries to calculate over the scores of multiple replications

    Returns: The rows to be added to the result dataframe

    """
    # TODO: Rewrite to use dict with respective names and values
    dict_of_results = {
        format_metric(metric, form): form(scores_df[metric])
        for metric in scores_df.columns
        for form in formats
    }
    return dict_of_results
