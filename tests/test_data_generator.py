from justcause.data.generators.ihdp import multi_expo_on_ihdp


def test_ihdp_generator():
    ihdp = multi_expo_on_ihdp(setting="multi-modal", replications=10)
    assert len(ihdp.data) == 747 * 10
    assert len(ihdp.data.groupby("rep")) == 10

    ihdp = multi_expo_on_ihdp(setting="multi-modal", size=500, replications=10)
    assert len(ihdp.data) == 500 * 10

    ihdp = multi_expo_on_ihdp(setting="exponential", size=200, replications=100)
    assert len(ihdp.data) == 200 * 100
