A work in progress for causal estimator evaluation. The framework aims to make comparison of
methods easier, by allowing to compare them across both generated and existing datasets.

# General Structure
## Methods
`CausalMethod` provides the abstract class which the specific implementation have to inherit.
Essentially, to add a new method, one has to overwrite its `fit()` and `predict()` functions.
Consider `causaleval/methods/causal_forest.py` for an example that uses `rpy2` to wrap an existing method
or see `causaleval/methods/basics/outcome_regression.py` for a simple, pure-python implementation.

## Datasets
`DataProvider` is the abstract interface to a new dataset. In the class, most of the functionality for computing
train/test splits and epoch generators is already provided. Thus, to add a new dataset, one has to overwrite only
`load_training_data()` and set the class attributes for `x, y, t, y_1, y_0, y_cf`. Based on that data, `DataProvider`
will generate a train/test split and return training data of requested sizes if available.

See `causaleval/data/sets/ihdp.py` for a few examples. What is special, `IHDPRelicaProvider` will
return a different one of 1000 replications every time `get_training_data()` is called, thus the method has been
overriden from `DataProvider` to accomodate that functionality.

See `causaleval/data/generators/acic.py` for a sophisticated parametric generation that is also


# To Run Yourself
## Locate Datafiles
Put the data files as required by the provider into a directory `eval/datasets/...`.
E.g. the IHDP data is in `eval/datasets/ihdp/csv` and add add a relative path to the config.py starting
form the content root / git root. E.g. Add relative path to ihdp `datasets/ihdp/csv`. Join that relative path
with the dynamically determined `ROOT_PATH` via `os.path.join()`

See `config.py` for possible configuration of data locations.

## Create Folders and Environments
Create the folder `results` and `results/plot` to store the outputs via the command
```bash
mkdir -p results/plots
```

Setup the environments for Python and R as explained below.

## Configure Experiment
Configure the experiment you want to run in `experiment.py`

 - Choose which dataset to run by putting the respective `DataProvider` instance
   in the `datasets` array
 - Choose which methods to run by putting the `CausalMethod` instances in the `methods` array
 - Define the sizes you want to evaluate
 - Define the number of replications via the `num_runs` parameter of `StandardEvaluation`.

# Setup Sacred for logging
To use `sacred` with the MongoObserver, you have to install `mongod` first.

## MongoDB
Run `mkdir sacred` in the project root.
Then, once `mongod` is installed, create a local mongoDB instance by running
```bash
mongod --db-path sacred
```
Configure the Observer in `experiment.py` to connect to your local host

## Omniboard
Install and run Omniboard with the default settings and it will connect to your running `mongod` instance.
After that, the dashboard is available at `localhost:9000`.

## Alternative
When running on a cluster, or when you don't want to setup MongoDB, you can use the `FileStorageObserver` instead of
the `MongoObserver`.
Simply create directory `my_runs` and run the experiment via
```bash
python causaleval/experiment.py -F my_runs
```

## What is logged
Results are stored row wise in a csv with fields:
`Metric, Method, Dataset, Size, Sample, Time, Score, Std`. Meaning
 - Metric used (e.g. PEHE, MSE on ATE, ...) + the number of replications (e.g. PEHE100)
 - Method evaluated
 - Dataset used
 - Size of the training sample
 - Evaluated on Test or Train
 - Time required for computation
 - Score of the Metric on given method, dataset, size, sample reported as mean over all replications
 - Std error over all replications


# Setting up the environment
## In Conda
Using the conda environment specification file `conda_env.txt` you can setup a conda environment with Python and R
installed:
```bash
conda create --name new_env --file conda_env.txt
```

## R - 3.6
 You might have to install a few R packages from GitHub sources manually. Namely,
 - https://github.com/saberpowers/causalLearning
 - https://github.com/xnie/rlearner

 via the commands

```R
install.packages('devtools')
library(devtools)
install_github('saberpowers/causalLearning')
install_github('xnie/rlearner')
```

within a local R environment. Within that R command, execute `.libPaths()` to get the path to the
corresponding R_HOME, which you need to put in the `config.py` in order for `rpy2` to use that R
directory with the pre-installed packages

In order for the build process to work you might have to source the following or add it to your `bashrc`:

```bash
export LC_ALL="en_US.UTF-8"
```

## Python 3.7.1
Install the packages via `pip install -r requirements.txt` within your virtualenv or conda environment

Setup the `PYTHONPATH` variable to know the module, so that the imports work:

```bash
export PYTHONPATH=${PYTHONPATH}:/causaleval
```
from the root directory of this repository.

# Further Work
Some steps to continue the work on this project would be
  - Rewrite the plot functions in `utils.py` to simply take `DataProvider` as inputs and handle the internals within
    the functions.
  - Implement a run-checker that ensures that all methods fit on the data and/or that no complications arise,
    before expensive computation is started.
    (e.g. requested size is to big for given DataProvider)
  - Obviously, add more methods and reference datasets
  - Write tests ;)


