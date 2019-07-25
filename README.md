# Locate Datafiles
Put the data files as required by the provider into a directory `eval/datasets/...`.
E.g. the IHDP data is in `eval/datasets/ihdp/csv` and add add a relative path to the config.py starting
form the content root / git root. E.g. Add relative path to ihdp `datasets/ihdp/csv`. Join that relative path
with the dynamically determined `ROOT_PATH` via `os.path.join()`



