import numpy as np
import pandas as pd


def get_train_test(
    data_bunch,
    train_size=0.8,
    same_split_in_replication=False,
    separate_replications=False,
    random_seed=None,
):
    """

    Args:
        data_bunch:
        train_size:
        same_split_in_replication:
        separate_replications: TODO: implement a split of train/test across replications
        random_seed: TODO: implement settable random seed for sampling

    Returns: (train, test) tuple

    """
    if type(data_bunch) is pd.DataFrame:
        df = data_bunch
    else:
        df = data_bunch.data

    use_test = "has_test" in data_bunch and data_bunch.has_test is True
    if use_test:
        return df.loc[~df["test"]], df.loc[df["test"]]

    if "size" in df:
        use_varying_size = True
        num_instances = int(df["size"].min())
        same_split_in_replication = False
    else:
        use_varying_size = False
        num_instances = len(df.groupby("sample_id"))

    num_test = int(num_instances * (1 - train_size))

    # Create test column
    df["test"] = False

    test_idxs = np.random.choice(range(num_instances), size=num_test)

    for rep in df["rep"].unique()[:-1]:
        # For each replication set train/test
        if not same_split_in_replication:
            if use_varying_size:
                size = int(df.loc[df["rep"] == rep]["size"].unique())
            else:
                size = int(num_test)

        test_idxs = np.random.choice(range(size), size=int(size * (1 - train_size)))
        bool_array = np.repeat(False, size)
        bool_array[test_idxs] = True
        rep_df = df.loc[df["rep"] == rep]
        rep_df.loc[:, "test"] = bool_array
        df.loc[df["rep"] == rep] = rep_df

    return df.loc[~df["test"]], df.loc[df["test"]]
