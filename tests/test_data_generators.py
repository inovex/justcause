import pytest

import numpy as np

from justcause.data.generators.ihdp import multi_expo_on_ihdp
from justcause.data.generators.rlearner import rlearner_simulation_data
from justcause.data.generators.toy import toy_data_emcs, toy_data_synthetic
from justcause.data.sets.ihdp import get_ihdp_covariates
from justcause.data.utils import generate_data


def test_generator():
    ihdp_cov = get_ihdp_covariates()

    def treatment(covariates, *, random_state, **kwargs):
        return np.ones(len(covariates), dtype=np.int)

    def outcomes(covariates, *, random_state, **kwargs):
        y_0 = np.zeros(len(covariates), dtype=np.int)
        y_1 = np.ones(len(covariates), dtype=np.int)
        mu_0, mu_1 = y_1, y_0
        return mu_0, mu_1, y_0, y_1

    def covariates(_, *, random_state, **kwargs):
        return ihdp_cov

    gen = generate_data(covariates=ihdp_cov, treatment=treatment, outcomes=outcomes)
    df = list(gen)[0]
    assert len(df) == len(ihdp_cov)

    with pytest.raises(AssertionError):
        generate_data(covariates=covariates, treatment=treatment, outcomes=outcomes)

    gen = generate_data(
        covariates=covariates,
        treatment=treatment,
        outcomes=outcomes,
        n_samples=len(ihdp_cov),
    )
    df = list(gen)[0]
    assert len(df) == len(ihdp_cov)
    assert len(set(df.columns).intersection({f"x_{i}" for i in range(25)})) == 25

    gen = generate_data(
        covariates=ihdp_cov.to_numpy(),
        treatment=treatment,
        outcomes=outcomes,
        n_samples=100,
    )
    df = list(gen)[0]
    assert len(df) == 100
    assert len(set(df.columns).intersection({f"x{i}" for i in range(25)})) == 25

    cov_names = [f"cov{i}" for i in range(25)]
    gen = generate_data(
        covariates=ihdp_cov.to_numpy(),
        treatment=treatment,
        outcomes=outcomes,
        covariate_names=cov_names,
    )
    df = list(gen)[0]
    assert len(df) == len(ihdp_cov)
    assert len(set(df.columns).intersection(set(cov_names))) == 25

    # Works with DataFrame and np.array when n_samples is given
    gen = generate_data(
        covariates=ihdp_cov, treatment=treatment, outcomes=outcomes, n_samples=500,
    )
    df = list(gen)[0]
    assert len(df) == 500
    assert len(set(df.columns).intersection({f"x_{i}" for i in range(25)})) == 25


def test_ihdp_generator():
    gen = multi_expo_on_ihdp(setting="multi-modal", n_replications=10)
    ihdp_reps = list(gen)
    assert len(ihdp_reps[0]) == 747
    assert len(ihdp_reps) == 10

    gen = multi_expo_on_ihdp(setting="multi-modal", n_samples=500, n_replications=10)
    ihdp_reps = list(gen)
    assert len(ihdp_reps) == 10
    assert len(ihdp_reps[0]) == 500

    gen = multi_expo_on_ihdp(setting="exponential", n_samples=200, n_replications=100)
    ihdp_reps = list(gen)
    assert len(ihdp_reps) == 100
    assert len(ihdp_reps[0]) == 200


def test_toy_generator():
    n_samples = 10000
    n_replications = 5

    gen = toy_data_synthetic(
        setting="simple", n_samples=n_samples, n_replications=n_replications
    )
    toy_reps = list(gen)
    assert len(toy_reps) == n_replications
    assert len(toy_reps[0]) == n_samples

    n_samples = 100
    n_replications = 50

    toy_reps = list(
        toy_data_synthetic(
            setting="hard", n_samples=n_samples, n_replications=n_replications
        )
    )
    assert len(toy_reps[0]) == n_samples
    assert len(toy_reps) == n_replications

    n_samples = 10000
    n_replications = 10

    toy_reps = list(
        toy_data_emcs(
            setting="simple", n_samples=n_samples, n_replications=n_replications
        )
    )
    assert len(toy_reps) == n_replications
    assert len(toy_reps[0]) == n_samples

    toy_reps = list(
        toy_data_emcs(
            setting="hard", n_samples=n_samples, n_replications=n_replications
        )
    )
    assert len(toy_reps) == n_replications
    assert len(toy_reps[0]) == n_samples

    with pytest.raises(RuntimeError):
        toy_data_emcs(
            setting="non-existent", n_samples=n_samples, n_replications=n_replications
        )


def test_rlearner_simulation():
    n_samples = 1000
    n_covariates = 7
    n_rep = 100

    gen = rlearner_simulation_data(n_samples, n_covariates, n_rep, setting="A")
    data = list(gen)
    assert len(data) == n_rep
    assert len(data[0]) == n_samples

    gen = rlearner_simulation_data(n_samples, n_covariates, n_rep, setting="B")
    data = list(gen)
    assert len(data) == n_rep
    assert len(data[0]) == n_samples

    gen = rlearner_simulation_data(n_samples, n_covariates, n_rep, setting="C")
    data = list(gen)
    assert len(data) == n_rep
    assert len(data[0]) == n_samples
    assert np.mean(data[0].mu_1 - data[0].mu_0) == 1

    gen = rlearner_simulation_data(n_samples, n_covariates, n_rep, setting="D")
    data = list(gen)
    assert len(data) == n_rep
    assert len(data[0]) == n_samples
