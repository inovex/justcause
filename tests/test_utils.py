from justcause.data.utils import iter_rep


def test_iter_rep(ihdp_data):
    single_rep = next(iter_rep(ihdp_data.data))
    assert single_rep["rep"].unique().item() == 0
