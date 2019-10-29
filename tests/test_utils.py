from justcause.data.utils import iter_rep


def test_iter_rep(dummy_df):
    assert "rep" in dummy_df.columns
    single_rep = next(iter_rep(dummy_df))
    assert "rep" not in single_rep.columns
    assert single_rep.shape[0] == 5
