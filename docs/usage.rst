=====
Usage
=====

This sections gives you a detailed explanation how to use JustCause.

Data Sets
=========

Mention here concepts of replications, mention the structure etc.

Handling Data
=============

JustCause uses a generalization of a Pandas :class:`~pandas.DataFrame` for managing your data named :class:`~.CausalFrame`.
A CausalFrame encompasses all the functionality of a Pandas DataFrame but additionally keeps track which columns
are a *covariates*, *treatment*, *outcome* or *others*. This allows to easily access them in a programmatic way.

All data sets provided by JustCause are provided as lists of CausalFrames, i.e. for each replication one CausalFrame.
Thus, we get a single CausalFrame ``cf`` from one of the provided data sets by::

    >>> from justcause.data.sets import load_ihdp
    >>>
    >>> cf = load_ihdp(select_rep=0)[0]  # select replication 0
    >>> type(cf)
    justcause.data.frames.CausalFrame

As usual, ``cf.columns`` would list the names of all columns. To find out if a column actually is a covariate, treatment,
outcome or something else, we can use the attribute accessor ``names``::

    >>> cf.names.treatment
    't'
    >>> cf.names.outcome
    'y'
    >>> cf.names.covariates
    ['0',
     '1',
     '2',
     '3',
     ...
     '21',
     '22',
     '23',
     '24']
    >>> cf.names.others
    ['sample_id', 'mu_1', 'mu_0', 'y_cf', 'ite']

This allows us to easily apply transformations for instance only to covariates. In general, this leads to more robust code
since the API of a CausalFrame enforces the differentiation between covariates, outcome, treatment and other columns
such as metadata like a datetime or an id of an observation.

If we want to construct a CausalFrame, we do that just in the same way as with a DataFrame but have to specify covariate,
treatment and outcome columns::

    >>> import justcause as jc
    >>> from numpy.random import rand, randint
    >>> import pandas as pd
    >>>
    >>> N = 10
    >>> dates = pd.date_range('2020-01-01', periods=N)
    >>> cf = jc.CausalFrame({'c1': rand(N),
    >>>                      'c2': rand(N),
    >>>                      'date': dates,
    >>>                      't': randint(2, size=N),
    >>>                      'y': rand(N)
    >>>                      },
    >>>                      covariates=['c1', 'c2'],
    >>>                      treatment='t',
    >>>                      outcome='y')

In our example, we do not need to pass ``treatment='t'`` and ``outcome='y'`` since ``'t'`` and ``'y'`` are used as default
values for the parameters ``treatment`` and ``outcome``, respectively, if they exist as column names.
All columns not listed as covariates, treatment and outcome will be considered as *others*::

    >>> cf.names.others
    ['date']

Working with Learners
=====================

Within the PyData stack, `Numpy`_ surely is the lowest common denominator and is thus used by a lot of libraries. Since
JustCause mainly wraps third-party libraries for causal methods under a common API, the decision was taken to only allow
passing Numpy arrays to the learners, i.e. causal methods, within JustCause. This allows for more flexibility and keeps
the abstraction layer to the original method much smaller.

The ``fit`` method of a learner takes at least the parameters ``X`` for the covariate matrix,  ``t`` for the treatment
and ``y`` for the outcome, i.e. target, vector as Numpy arrays. In order to bridge the gap between rich CausalFrames and
plain arrays, a :class:`~.CausalFrame` provides the attribute accessor ``np`` (for *numpy*). Using it, we can easily pass
the covariates, treatment and outcome to a learner::

    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> reg = RandomForestRegressor()
    >>> learner = jc.learners.SLearner(reg)
    >>> learner.fit(cf.np.X, cf.np.t, cf.np.y)



.. _Numpy: https://numpy.org/


