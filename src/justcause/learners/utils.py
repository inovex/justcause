from typing import List

from causalml.propensity import ElasticNetPropensityModel
from sklearn.calibration import CalibratedClassifierCV

def replace_factual_outcomes(y_0, y_1, y, t):
    """ Replaces the predicted components with factual observations where possible

    Args:
        y_0: predicted control outcomes
        y_1: predicted treatment outcomes
        y: factual outcomes
        t: factual treatment indicators

    Returns: y_0, y_1 with factual outcomes replaced where possible
    """
    for i in range(len(t)):
        if t[i] == 1:
            y_1[i] = y[i]
        else:
            y_0[i] = y[i]
    return y_0, y_1


def install_r_packages(package_names: List[str], verbose=False):
    """ Installs the packages if needed using rpy2 utility functions"""
    import rpy2.robjects.packages as rpackages
    from rpy2 import robjects
    from rpy2.robjects import StrVector

    robjects.r.options(download_file_method="curl")
    utils = rpackages.importr("utils")
    utils.chooseCRANmirror(ind=1)

    names_to_install = [x for x in package_names if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install), verbose=verbose)


# TODO: Consider https://scikit-learn.org/stable/modules/calibration.html
#  for a discussion of setting proper probability estimatators
def set_propensity_learner(p_learner=None):
    """ Returns default propensity learner or checks the given p_learner"""
    if p_learner is None:
        return ElasticNetPropensityModel()
    else:
        assert hasattr(
            p_learner, "predict_proba"
        ), "propensity learner must have predict_proba method"

        return CalibratedClassifierCV(p_learner)


# TODO: This is really only required when the ElasticNetPropensityModel is used
def fit_predict_propensity(p_learner, x, t):
    """ Check which model is used and predict probability accordingly"""
    if isinstance(p_learner, ElasticNetPropensityModel):
        return p_learner.fit_predict(x, t)
    else:
        p_learner.fit(x, t)
        return p_learner.predict_proba(x)[:, 1]
