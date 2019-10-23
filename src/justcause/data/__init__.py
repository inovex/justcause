from pathlib import Path

#: URL for retrieving the datasets
DATA_URL = "https://raw.github.com/inovex/justcause-data/master/"
#: Directory for storing the datasets locally
DATA_DIR = Path("~/.justcause_data").expanduser()
#: Default columns besides the covariates in each dataframe
DATA_COLS = ["t", "y", "y_cf", "y_0", "y_1", "ite", "rep", "sample_id"]
