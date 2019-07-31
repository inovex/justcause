# Locate Datafiles
Put the data files as required by the provider into a directory `eval/datasets/...`.
E.g. the IHDP data is in `eval/datasets/ihdp/csv` and add add a relative path to the config.py starting
form the content root / git root. E.g. Add relative path to ihdp `datasets/ihdp/csv`. Join that relative path
with the dynamically determined `ROOT_PATH` via `os.path.join()`



# On logging
Results are stored row wise in a csv with fields: `Metric, Method, Dataset, Score`.

Since it is possible to evaluate a dataset on different sizes of training etc. we will later
add identifiers to the dataset name like `IHDP-1k` for methods trained on 1000 instances of IHDP.

# Notes no DataProviders
 - Files are in np.array format when they are retrieved from the DataProvider