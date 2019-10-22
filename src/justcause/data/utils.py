import pandas as pd
from sklearn.model_selection import train_test_split


def group_split(group, train_size, random_state):
    train, test = train_test_split(
        group, train_size=train_size, random_state=random_state
    )
    train.loc[:, "test"] = False
    test.loc[:, "test"] = True
    return pd.concat([train, test]).sort_values("sample_id")


def get_train_test(data_bunch, train_size=0.8, random_state=None):
    """ Applies a train_test_split on each replication

    Args:
        data_bunch: data bunch or dataframe containing the dataset
        train_size: between 0 and 1, indicating the ratio of train/test in each section
        random_state: random seed for train_test_split

    Returns: (train, test) tuple

    """
    if type(data_bunch) is pd.DataFrame:
        df = data_bunch
    else:
        df = data_bunch.data

    use_test = "has_test" in data_bunch and data_bunch.has_test is True
    if use_test:
        return df.loc[~df["test"]], df.loc[df["test"]]

    df = (
        df.groupby("rep")
        .apply(group_split, train_size=0.8, random_state=random_state)
        .reset_index(drop=True)
    )
    return df.loc[~df["test"]], df.loc[df["test"]]
