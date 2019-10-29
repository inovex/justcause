from justcause.data.generators.ihdp import multi_expo_on_ihdp
from justcause.data.generators.toy import toy_data_synthetic, toy_example_emcs


def test_ihdp_generator():
    ihdp = multi_expo_on_ihdp(setting="multi-modal", replications=10)
    assert len(ihdp.data) == 747 * 10
    assert len(ihdp.data.groupby("rep")) == 10

    ihdp = multi_expo_on_ihdp(setting="multi-modal", size=500, replications=10)
    assert len(ihdp.data) == 500 * 10

    ihdp = multi_expo_on_ihdp(setting="exponential", size=200, replications=100)
    assert len(ihdp.data) == 200 * 100


def test_toy_generator():
    size = 10000
    replications = 5

    toy = toy_data_synthetic(setting="simple", size=size, replications=replications)
    assert len(toy.data) == size * replications
    assert len(toy.data.groupby("rep")) == replications

    size = 100
    replications = 50

    toy = toy_data_synthetic(setting="hard", size=size, replications=replications)
    assert len(toy.data) == size * replications
    assert len(toy.data.groupby("rep")) == replications

    size = 10000
    replications = 10

    toy = toy_example_emcs(setting="simple", size=size, replications=replications)
    assert len(toy.data) == size * replications
    assert len(toy.data.groupby("rep")) == replications
