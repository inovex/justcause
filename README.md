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

<br/>

# Introduction

Evaluating causal inference methods in a scientifically thorough way is a cumbersome and error-prone task.
To foster good scientific practice **JustCause** provides a framework to easily:

1. evaluate your method using common data sets like IHDP, IBM ACIC, and others;
2. create synthetic data sets with a generic but standardized approach;
3. benchmark your method against several baseline and state-of-the-art methods.

Our *cause* is to develop a framework that allows you to compare methods for causal inference
in a fair and *just* way. JustCause is a work in progress and new contributors are always welcome.

# Installation

If you just want to use the functionality of JustCause, install it with:
```
pip install justcause
```
Consider using [conda] to create a virtual environment first.

Developers that want to develop and contribute own algorithms and data sets to the JustCause framework, should:

1. clone the repository and change into the directory
   ```
   git clone https://github.com/inovex/justcause.git
   cd justcause
   ```

2. create an environment `justcause` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
3. activate the new environment with
   ```
   conda activate justcause
   ```
4. install `justcause` with:
   ```
   python setup.py install # or `develop`
   ```

Optional and needed only once after `git clone`:

5. install several [pre-commit] git hooks with:
   ```
   pre-commit install
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.


# Related Projects & Resources

 1. [causalml]: causal inference with machine learning algorithms in Python
 2. [DoWhy]: causal inference using graphs for identification
 3. [EconML]: Heterogeneous Effect Estimation in Python
 4. [awesome-list]: A very extensive list of causal methods and respective code
 5. [IBM-Causal-Inference-Benchmarking-Framework]: Causal Inference Benchmarking Framework by IBM
 6. [CausalNex]: Bayesian Networks to combine machine learning and domain expertise for causal reasoning.

## Note

This project has been set up using [PyScaffold] 3.2.2 and the [dsproject extension] 0.4.
For details and usage information on PyScaffold see https://pyscaffold.org/.


[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
[causalml]: https://github.com/uber/causalml
[DoWhy]: https://github.com/Microsoft/dowhy
[EconML]: https://github.com/microsoft/EconML
[awesome-list]: https://github.com/rguo12/awesome-causality-algorithms
[IBM-Causal-Inference-Benchmarking-Framework]: https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework
[CausalNex]: https://causalnex.readthedocs.io/
