[![Docs Status](https://readthedocs.org/projects/justcause/badge/?version=latest)](https://justcause.readthedocs.io/en/latest/?badge=latest)
[![CI Status](https://api.cirrus-ci.com/github/inovex/justcause.svg?branch=master)](https://cirrus-ci.com/github/inovex/justcause)
[![Coverage Status](https://coveralls.io/repos/github/inovex/justcause/badge.svg?branch=master)](https://coveralls.io/github/inovex/justcause?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)
[![PyPI-Server](https://img.shields.io/pypi/v/justcause.svg)](https://pypi.org/project/justcause/)


# JustCause

Comparing algorithms for causality analysis in a fair and just way.

## Description

A **work in progress** for causal estimator evaluation. The framework aims to make comparison of
methods easier, by allowing to compare them across both generated and existing datasets.

#### ToDos:

* Get rid of all those unused variables, currently suppressed with `# noqa: F841`
* The package itself should not depend on sacred, only the experiment a user of the package sets up. Thus remove it from metrics.py
* Separate Loggin/Writing stuff from the actual calculation of metrics in metrics.py (single level of responsibility)
* make the package itself independent of Sacred, just advocate it as best practice
* migrate all files ending in `-old` and delete them if no longer necessary
* create some proper unittests and use pytest instead of the shipped unittest
* add the final bachelor thesis as pdf under `docs` and reference it in Sphinx
* use Sphinx (checkout `docs` folder) to create command reference and some explanations.
* remove `configs/config.py` by passing only relevant information as arguments to the functions of the package. Configuration
  of an experiment is subject to the experiment itself.
* adhere to `pep8` and other standards. Use `pre-commit` (which is set up below) to check and correct all mistakes
* Don't fix things like random seed within the package, it's a library, advocate to do this outside
 (name this best-practice within the docs)
* separate modules that only do math from plotting modules. Why would the generators/acic module need matplotlib as dependency
* follow import order, first Python internal modules, then external, then the modules of your package.
* use PyCharm and check for the curly yellow underline hints how to improve the code
* add some example notebooks in the notebooks folder
* add the libraries which a required (no visualisation) into setup.cfg under requires.
* Check licences of third-party methods and add and note them accordingly. Within the __init__.py of the subpackage
  add a docstring and state the licences and the original authors.
* Do not set environment variables inside library, rather state this somewhere in the docs. os.environ['L_ALL']
* Never print something in a library, use the logging module for logging. Takes a while to comprehend
* move the `experiment.py` module into the `scripts` folder because it's actually using the package (fix the imports accordingly)
* avoid plotting to `results/plots/S-Learner - LinearRegressionrobustness.png'` in the unittests (right now the directory
  needs to be created for the unittests to run)
* Do imports within functions only when really necessary (there are rare cases only) otherwise on the top of the module
* Don't set `R_HOME` environment variable and rely on what conda is doing for you. Avoid setting any kind of path
  via environment variables.
* Remove all `if __name__ == "__main__":` sections from the modules in the justcause package
* When files are downloaded keep them under `~/.justcause` (, i.e. hidden directory in the home dir) and access them.
  Check out how this in done under https://github.com/maciejkula/spotlight/blob/master/spotlight/datasets/_transport.py
* Use cirrus as CI system.
* Consider using the abstract base classes of Scikit-Learn with their Mixin concept instead of providing an own.

## Installation

In order to set up the necessary environment:

1. create an environment `justcause` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate justcause
   ```
3. install `justcause` with:
   ```
   python setup.py install # or `develop`
   ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```
   pre-commit install
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

5. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yaml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yaml` for the exact reproduction of your
   environment with:
   ```
   conda env export -n justcause -f environment.lock.yaml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yaml` using:
   ```
   conda env update -f environment.lock.yaml --prune
   ```
## Project Organization

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.s
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── justcause           <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## Note

This project has been set up using PyScaffold 3.2.2 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
