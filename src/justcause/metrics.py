"""Contains metrics to score ITE predictions against the true ITEs"""
import numpy as np


def pehe_score(true: np.array, predicted: np.array) -> float:
    r"""Calculates the root of PEHE score

    PEHE: Precision in Estimation of Heterogeneous Effects
    Essentially, the root PEHE score is a root mean squared error (RMSE)

    .. math::
            \epsilon_{P E H E} &= n^{-1} \sum_{i=1}^n \left(\big[Y_i(1)-Y_i(0)\big] -
            \left[\hat{Y}_i(1)-\hat{Y}_i(0)\right]\right)^{2}  \\
            &= n^{-1} \sum_{i=1}^n \left(  \hat{\tau}(x_i) - \tau(x_i)\right)^{2}.

    Args:
        true: true ITE scores
        predicted: predicted ITE scores

    References:
        [1] “Bayesian Nonparametric Modeling for Causal Inference,”
            J. Hill, J. Comput. Graph. Stat., vol. 20, no. 1, pp. 217–240, 2011.

    Returns:
        scalar root PEHE score
    """
    return np.sqrt(np.mean(np.square(predicted - true)))


def mean_absolute(true: np.array, predicted: np.array) -> float:
    r"""Calculates the absolute mean error

    .. math::
        \epsilon_{ATE} =  \vert \tau - \hat{\tau} \vert,

    Args:
        true: true ITEs
        predicted: predicted ITEs

    """
    return np.abs(np.mean(true) - np.mean(predicted))


def enormse(true: np.array, predicted: np.array) -> float:
    r"""Calculates the Effect NOrmalised Root Means Squared Error (ENoRMSE)

    .. math::
        \epsilon_{E N O R M S E} = \sqrt{n^{-1} \sum_i^n
            (1 - \frac{\hat{\tau}(x_i)}{\tau(x_i)})^2}.

    References:
        [1] “Benchmarking Framework for Performance-Evaluation
            of Causal Inference Analysis,” Y. Shimoni, C. Yanover,
            E. Karavani, and Y. Goldschmnidt; 2018

    Args:
        true: true ITE
        predicted: predicted ITE

    """
    return np.sqrt(
        np.sum(np.power((1 - (predicted + 0.0001) / (true + 0.0001)), 2))
        / true.shape[0]
    )


def bias(true: np.array, predicted: np.array) -> float:
    r"""Calculates the mean bias of the prediction

    .. math::
        \epsilon_{B I A S} = n^{-1} \sum_i^n (\hat{\tau}(x_i) - \tau(x_i)).

    Args:
        true: true ITE
        predicted: predicted ITE

    """
    return np.sum(predicted - true) / true.shape[0]
