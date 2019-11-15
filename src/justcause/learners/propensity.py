from causalml.propensity import ElasticNetPropensityModel
from pygam import LogisticGAM, s
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def get_default_estimator():
    """ Returns a default estimator"""
    return LogisticRegression()


def estimate_propensities(x, t):
    """ Estimates and calibrates propensity score using the default estimator and GAM"""
    learner = get_default_estimator()
    learner.fit(x, t)
    return calibrate(learner.predict_proba(x)[:, 1], t)


# TODO: Consider https://scikit-learn.org/stable/modules/calibration.html
#  for a discussion of setting proper probability estimatators
def set_propensity_learner(p_learner=None):
    """ Returns default propensity learner or checks the given p_learner"""
    if p_learner is None:
        return get_default_estimator()
    else:
        assert hasattr(
            p_learner, "predict_proba"
        ), "propensity learner must have predict_proba method"

        return CalibratedClassifierCV(p_learner, method="isotonic")


def calibrate(propensities, treatment):
    """
    Post-hoc calibration of propensity scores given the true treatments

    Args:
        propensities: propensity scores
        treatment: treatment indicator

    Returns: calibrated probabilites of treatment
    """
    gam = LogisticGAM(s(0)).fit(propensities, treatment)
    return gam.predict_proba(propensities)


# TODO: This is really only required when the ElasticNetPropensityModel is used
def fit_predict_propensity(p_learner, x, t):
    """ Check which model is used and predict probability accordingly"""
    if isinstance(p_learner, ElasticNetPropensityModel):
        return p_learner.fit_predict(x, t)
    else:
        p_learner.fit(x, t)
        return p_learner.predict_proba(x)[:, 1]
