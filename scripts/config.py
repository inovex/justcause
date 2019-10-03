# TODO: REMOVE THIS FILE, IT'S ONLY HERE TO MAKE experiment.py WORK
#       UNTIL justcause DOESN'T IMPORT IT ANYMORE EVERYWHERE.

import os

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # this is the git root
RELATIVE_IHDP_PATH = "datasets/ihdp/csv"
IHDP_PATH = os.path.join(ROOT_DIR, RELATIVE_IHDP_PATH)

RELATIVE_IHDP_REPLICA_PATH = "datasets/ihdp_from_r/A"
IHDP_REPLICA_PATH = os.path.join(ROOT_DIR, RELATIVE_IHDP_REPLICA_PATH)

RELATIVE_IHDP_REPLICA_PATH_SETTING_B = "datasets/ihdp_from_r/B"
IHDP_REPLICA_PATH_SETTING_B = os.path.join(
    ROOT_DIR, RELATIVE_IHDP_REPLICA_PATH_SETTING_B
)

RELATIVE_TWINS_PATH = "datasets/twins"
TWINS_PATH = os.path.join(ROOT_DIR, RELATIVE_TWINS_PATH)

RELATIVE_IBM_PATH = "datasets/ibm/scaling"
IBM_PATH = os.path.join(ROOT_DIR, RELATIVE_IBM_PATH)
RELATIVE_IBM_PATH_ROOT = "datasets/ibm"
IBM_PATH_ROOT = os.path.join(ROOT_DIR, RELATIVE_IBM_PATH_ROOT)

DB_NAME = "sacred"
DB_URL = "localhost:27017"

OUTPUT_PATH = os.path.join(ROOT_DIR, "results") + "/"
RESULT_PLOT_PATH = os.path.join(OUTPUT_PATH, "plots")

GENERATE_PATH = os.path.join(ROOT_DIR, "datasets/generated/")

ACIC_USE_COVARIATES = [
    "sex",
    "md_route",
    "dbwt",
    "estgest",
    "mbrace",
    "meduc",
    "precare",
]

ACIC_SELECTED_VALUES = ["sex", "md_route", "dbwt", "estgest", "mager41"]

# R Settings

# R_HOME = '/usr/local/Cellar/r/3.6.1/lib/R'

# ENV Settings

PLOT_WRITE = True
SERVER = False
LOG_FILE_PATH = os.path.join(ROOT_DIR, "results/plotlog")

# COLORS

CYAN = "#4ECDC4"
BLUE = "#59D2FE"
RED = "#FF6B6B"
YELLOW = "#FAA916"
GREY = "#4A6670"
COLOR_LIST = [CYAN, BLUE, RED, YELLOW, GREY]
