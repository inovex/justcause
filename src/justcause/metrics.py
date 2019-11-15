import numpy as np


def pehe_score(true: np.array, predicted: np.array) -> float:
    """ Calculates the root of PEHE score

    PEHE: Precision in Estimation of Heterogeneous Effects

    Essentially, the root PEHE score is a RMSE.

    Args:
        true: true ITE scores
        predicted: predicted ITE scores

    References:
        [1] “Bayesian Nonparametric Modeling for Causal Inference,”
            J. Hill, J. Comput. Graph. Stat., vol. 20, no. 1, pp. 217–240, 2011.
    Returns: the scalar root PEHE score
    """ ""
    return np.sqrt(np.mean(np.square(predicted - true)))


def absolute_mean(true: np.array, predicted: np.array) -> float:
    """ Calculates the absolute mean error"""
    return np.abs(np.mean(true) - np.mean(predicted))


def enormse(true: np.array, predicted: np.array) -> float:
    """
    Calculates the Effect NOrmalised Root Means Squared Error (ENoRMSE)

    References:
        [1] “Benchmarking Framework for Performance-Evaluation
            of Causal Inference Analysis,” Y. Shimoni, C. Yanover,
            E. Karavani, and Y. Goldschmnidt; 2018

    Args:
        true: true ITE scores
        predicted: predicted ITE scores

    Returns: the scalar ENoRMSE score

    """
    return np.sqrt(
        np.sum(np.power((1 - (predicted + 0.0001) / (true + 0.0001)), 2))
        / true.shape[0]
    )


def bias(true: np.array, predicted: np.array) -> float:
    """ Calculates the mean bias of the prediction"""
    return np.sum(predicted - true) / true.shape[0]
