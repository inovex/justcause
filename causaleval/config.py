import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # this is the git root
RELATIVE_IHDP_PATH = 'datasets/ihdp/csv'
IHDP_PATH = os.path.join(ROOT_DIR, RELATIVE_IHDP_PATH)

RELATIVE_TWINS_PATH = 'datasets/twins'
TWINS_PATH = os.path.join(ROOT_DIR, RELATIVE_TWINS_PATH)

DB_NAME = 'sacred'
DB_URL = 'localhost:27000'
