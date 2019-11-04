import pytest

import numpy as np

from justcause.data.generators.ihdp import multi_expo_on_ihdp
from justcause.data.generators.toy import toy_data_synthetic, toy_example_emcs
from justcause.data.sets.ihdp import get_ihdp_covariates
from justcause.data.utils import generate_data


def test_generator():
    ihdp_cov = get_ihdp_covariates()

    def treatment(covariates):
        return np.ones(len(covariates), dtype=np.int)

    def outcomes(covariates):
        y_0 = np.zeros(len(covariates), dtype=np.int)
        y_1 = np.ones(len(covariates), dtype=np.int)
        mu_0, mu_1 = y_1, y_0
        return mu_0, mu_1, y_0, y_1

    def covariates(_):
        return ihdp_cov

    df = generate_data(covariates=ihdp_cov, treatment=treatment, outcomes=outcomes)
    assert len(df) == len(ihdp_cov)

    with pytest.raises(AssertionError):
        generate_data(covariates=covariates, treatment=treatment, outcomes=outcomes)

    df = generate_data(
        covariates=covariates,
        treatment=treatment,
        outcomes=outcomes,
        n_samples=len(ihdp_cov),
    )
    assert len(df) == len(ihdp_cov)
    assert len(set(df.columns).intersection({f"{i}" for i in range(25)})) == 25

    df = generate_data(
        covariates=ihdp_cov.to_numpy(),
        treatment=treatment,
        outcomes=outcomes,
        n_samples=100,
    )
    assert len(df) == 100
    assert len(set(df.columns).intersection({f"x{i}" for i in range(25)})) == 25

    cov_names = [f"cov{i}" for i in range(25)]
    df = generate_data(
        covariates=ihdp_cov.to_numpy(),
        treatment=treatment,
        outcomes=outcomes,
        covariate_names=cov_names,
    )
    assert len(df) == len(ihdp_cov)
    assert len(set(df.columns).intersection(set(cov_names))) == 25


def test_ihdp_generator():
    ihdp = multi_expo_on_ihdp(setting="multi-modal", n_replications=10)
    assert len(ihdp) == 747 * 10
    assert len(ihdp.groupby("rep")) == 10

    ihdp = multi_expo_on_ihdp(setting="multi-modal", n_samples=500, n_replications=10)
    assert len(ihdp) == 500 * 10

    ihdp = multi_expo_on_ihdp(setting="exponential", n_samples=200, n_replications=100)
    assert len(ihdp) == 200 * 100


def test_toy_generator():
    n_samples = 10000
    n_replications = 5

    toy = toy_data_synthetic(
        setting="simple", n_samples=n_samples, n_replications=n_replications
    )
    assert len(toy) == n_samples * n_replications
    assert len(toy.groupby("rep")) == n_replications

    n_samples = 100
    n_replications = 50

    toy = toy_data_synthetic(
        setting="hard", n_samples=n_samples, n_replications=n_replications
    )
    assert len(toy) == n_samples * n_replications
    assert len(toy.groupby("rep")) == n_replications

    n_samples = 10000
    n_replications = 10

    toy = toy_example_emcs(
        setting="simple", n_samples=n_samples, n_replications=n_replications
    )
    assert len(toy) == n_samples * n_replications
    assert len(toy.groupby("rep")) == n_replications

    toy = toy_example_emcs(
        setting="hard", n_samples=n_samples, n_replications=n_replications
    )
    assert len(toy) == n_samples * n_replications
    assert len(toy.groupby("rep")) == n_replications

    with pytest.raises(RuntimeError):
        toy_example_emcs(
            setting="non-existent", n_samples=n_samples, n_replications=n_replications
        )
