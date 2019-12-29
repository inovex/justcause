[![Docs Status](https://readthedocs.org/projects/justcause/badge/?version=latest)](https://justcause.readthedocs.io/en/latest/?badge=latest)
[![CI Status](https://api.cirrus-ci.com/github/inovex/justcause.svg?branch=master)](https://cirrus-ci.com/github/inovex/justcause)
[![Coverage Status](https://coveralls.io/repos/github/inovex/justcause/badge.svg?branch=master)](https://coveralls.io/github/inovex/justcause?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/en/stable/)
[![PyPI-Server](https://img.shields.io/pypi/v/justcause.svg)](https://pypi.org/project/justcause/)

<div style="text-align:center">
<p align="center">
<img alt="JustCause logo" src="https://justcause.readthedocs.io/en/latest/_static/logo.png">
</p>
</div>

# Introduction

Evaluating causal inference methods in a scientifically thorough way is a cumbersome and error-prone task.
To foster good scientific practice **JustCause** provides a framework to easily:

1. evaluate your method using common data sets like IHDP, IBM ACIC, and others;
2. create synthetic data sets with a generic but standardized approach;
3. benchmark your method against several baseline and state-of-the-art methods.

Our *cause* is to develop a framework that allows you to compare methods for causal inference
in a fair and *just* way. JustCause is a work in progress and new contributors are always welcome.

# Installation

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


# Further Work
Some steps to continue the work on this project would be:
  - Allow to use Confidence Intervals in the learners (see `causalml`, for example) and in the evaluation
  - Implement a fully parametric DGP, following the dimensions roughly outlined in Chapter 4 of the thesis
  - Rewrite the plot functions in `utils.py` to simply take `DataProvider` as inputs and handle the internals within
    the functions.
  - Implement a run-checker that ensures that all methods fit on the data and/or that no complications arise,
    before expensive computation is started.
    (e.g. requested size is to big for given DataProvider)
  - Obviously, add more methods and reference datasets ;)
  - use Sphinx (checkout `docs` folder) to create command reference and some explanations.
  - add some example notebooks in the `notebooks` folder of experiments and parameter studies
  - Check licences of third-party methods and add and note them accordingly. Within the __init__.py of the subpackage
    add a docstring and state the licences and the original authors.

## Note

This project has been set up using [PyScaffold] 3.2.2 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
