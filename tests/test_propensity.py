from justcause.learners.propensity import estimate_propensities, get_default_estimator


def test_default_learner(dummy_covariates_and_treatment):
    X, t = dummy_covariates_and_treatment
    default = get_default_estimator()
    # test that fit and predict_proba is available
    default.fit(X, t)
    p = default.predict_proba(X)[:, 1]
    assert p[0] > 0.9
    assert p[-1] < 0.1


def test_estimate(dummy_covariates_and_treatment):
    X, t = dummy_covariates_and_treatment
    p = estimate_propensities(X, t)
    assert p[0] > 0.9
    assert p[-1] < 0.1
