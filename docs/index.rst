============
Introduction
============

Evaluating causal inference methods in a scientifically thorough way is a cumbersome and error-prone task.
To foster good scientific practice **JustCause** provides a framework to easily:

1. evaluate your method using common data sets like IHDP, IBM ACIC, and others;
2. create synthetic data sets with a generic but standardized approach;
3. benchmark your method against several baseline and state-of-the-art methods.

Our *cause* is to develop a framework that allows you to compare methods for causal inference
in a fair and *just* way. JustCause is a work in progress and new contributors are always welcome.

The reasons for creating a library like JustCause are laid out in the thesis
:download:`A Systematic Review of Machine Learning Estimators for Causal Effects <thesis-mfranz.pdf>`
of Maximilian Franz. Therein, it is shown that many publications about causality:

    * lack reproducibility,
    * use different versions of the seemingly same data set,
    * fail to state that some theoretical conditions in the data set are not met,
    * miss several state of the art methods in their comparison.

A more standardised approach, as offered by JustCause, is able to improve these points.

Installation
============

Install JustCause with::

    pip install justcause

but consider using `conda`_ to set up an isolated environment beforehand. This can be done with::

    conda env create -f environment.yaml
    conda activate justcause

with the following :download:`environment.yaml <../environment.yaml>`.

Quickstart
==========

For a minimal example we are going to load the `IHDP`_ (Infant Health and Development Program) data set,
do a train/test split, apply a basic learner on each replication and display some metrics::

    >>> from justcause.data.sets import load_ihdp
    >>> from justcause.learners import SLearner
    >>> from justcause.learners.propensity import estimate_propensities
    >>> from justcause.metrics import pehe_score, mean_absolute
    >>> from justcause.evaluation import calc_scores

    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LinearRegression

    >>> import pandas as pd

    >>> replications = load_ihdp(select_rep=[0, 1, 2])
    >>> slearner = SLearner(LinearRegression())
    >>> metrics = [pehe_score, mean_absolute]
    >>> scores = []

    >>> for rep in replications:
    >>>    train, test = train_test_split(rep, train_size=0.8)
    >>>    p = estimate_propensities(train.np.X, train.np.t)
    >>>    slearner.fit(train.np.X, train.np.t, train.np.y, weights=1/p)
    >>>    pred_ite = slearner.predict_ite(test.np.X, test.np.t, test.np.y)
    >>>    scores.append(calc_scores(test.np.ite, pred_ite, metrics))

    >>> pd.DataFrame(scores)
       pehe_score  mean_absolute
    0    0.998388       0.149710
    1    0.790441       0.119423
    2    0.894113       0.151275

Contents
========

.. toctree::
   :maxdepth: 2

   Usage <usage>
   Best Practices <best_practices>
   License <license>
   Authors <authors>
   Changelog <changelog>
   Contributions & Help <contributing>
   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
.. _conda: https://docs.conda.io/
.. _IHDP: https://www.icpsr.umich.edu/web/HMCA/studies/9795
