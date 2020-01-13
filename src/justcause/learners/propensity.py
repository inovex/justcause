"""Default methods and helpers for estimation of propensities"""
from pygam import LogisticGAM, s
from sklearn.linear_model import LogisticRegression


def get_default_estimator():
    """Retrieve a default choice of probability estimator

    Returns:
        sklearn model with predict_proba() method
    """
    return LogisticRegression()


def estimate_propensities(X, t):
    """Estimates and calibrates propensity score using the default estimator and GAM

    See https://scikit-learn.org/stable/modules/calibration.html and the small study
    in `notebooks/probability_calibration.ipynb` for a more detailed discussion of
    probability calibration.

    Args:
        X: the covariate matrix of shape (num_samples, num_covariates)
        t: the binary treatment indicator of shape (num_samples)

    Returns:
        p: propensities calibrated on the covariates and treatment

    """
    learner = get_default_estimator()
    learner.fit(X, t)
    return calibrate_propensities(learner.predict_proba(X)[:, 1], t)


def calibrate_propensities(propensities, treatment):
    """Post-hoc calibration of propensity scores given the true treatments

    Args:
        propensities: propensity scores
        treatment: treatment indicator

    Returns:
        p: calibrated version of the propensities given
    """
    gam = LogisticGAM(s(0)).fit(propensities, treatment)
    return gam.predict_proba(propensities)
