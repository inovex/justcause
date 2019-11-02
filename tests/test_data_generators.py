from justcause.data.generators.ihdp import multi_expo_on_ihdp
from justcause.data.generators.toy import toy_data_synthetic, toy_example_emcs


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
