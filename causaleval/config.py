import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # this is the git root
RELATIVE_IHDP_PATH = 'datasets/ihdp/csv'
IHDP_PATH = os.path.join(ROOT_DIR, RELATIVE_IHDP_PATH)

RELATIVE_TWINS_PATH = 'datasets/twins'
TWINS_PATH = os.path.join(ROOT_DIR, RELATIVE_TWINS_PATH)

RELATIVE_IBM_PATH = 'datasets/ibm/scaling'
IBM_PATH = os.path.join(ROOT_DIR, RELATIVE_IBM_PATH)
RELATIVE_IBM_PATH_ROOT = 'datasets/ibm'
IBM_PATH_ROOT = os.path.join(ROOT_DIR, RELATIVE_IBM_PATH_ROOT)

DB_NAME = 'sacred'
DB_URL = 'localhost:27017'

OUTPUT_PATH = os.path.join(ROOT_DIR, 'results') + '/'

GENERATE_PATH = os.path.join(ROOT_DIR, 'datasets/generated/')