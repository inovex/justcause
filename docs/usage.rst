=====
Usage
=====

This sections gives you a detailed explanation how to use JustCause.

Handling Data
=============

JustCause uses a generalization of a Pandas :class:`~pandas.DataFrame` for managing your data named :class:`~.CausalFrame`.
A CausalFrame encompasses all the functionality of a Pandas DataFrame but additionally keeps track which columns
are a *covariates*, *treatment*, *outcome* or *others*. This allows to easily access them in a programmatic way.

All data sets provided by JustCause provided in terms of iterators over CausalFrames. Thus, we get a CausalFrame ``cf`` by::

    >>> from justcause.data.sets import load_ihdp

    >>> cf = load_ihdp()[0]  # select replication 0
    >>> type(cf)
    justcause.data.frames.CausalFrame

As usual ``cf.columns`` would list the column names. To find out which columns are covariates, treatment, outcome or others,
we can use the attribute accessor ``names``::

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

This allows us to easily apply transformations for instance only covariates and in general leads to more robust code
since the API of a CausalFrame enforces the differentiation between covariates, outcome, treatment and other columns
such as metadata like a datetime of an observation.

If we want to construct a CausalFrame, we do that just in the same way as with a DataFrame but have to define the roles
of the columns::

    >>> import justcause as jc
    >>> from numpy.random import rand, randint
    >>> import pandas as pd

    >>> N = 10
    >>> dates = pd.date_range('2020-01-01', periods=N)
    >>> cf = jc.CausalFrame({'c1': rand(N), 'c2': rand(N), 'date': dates, 't': randint(2, size=N), 'y': rand(N)},
                             covariates=['c1', 'c2'], treatment='t', outcome='y')

In our example, we do not need to pass ``treatment='t', outcome='y'`` since ``'t'`` and ``'y'`` are used as default
values for parameters ``treatment`` and ``outcome``, respectively if they exist as columns. All columns not listed as
covariates, treatment and outcome will be considered as *other*::

    >>> cf.names.other
    ['date']

Working with Learners
=====================

Within the PyData stack, `Numpy`_ surely is the lowest common denominator and is thus used by a lot of libraries. Since
JustCause only wraps third-party libraries for causal methods under a common API, the decision was taken to only allow
passing Numpy arrays to the learners, i.e. causal methods, within JustCause. This allows for more flexibility and keeps
the abstraction layer to the original method much smaller.

The ``fit`` method of a learner takes at least the parameters ``X`` for the covariate matrix,  ``t`` for the treatment
vector and ``y`` for the outcome, i.e. target, as Numpy arrays. In order to bridge the gap between rich CausalFrames and
plain arrays, a :class:`~.CausalFrame` provides the attribute accessor ``np`` (for ``numpy``). Using it, we can easily pass
the covariates, treatment and outcome to a learner::

    >>> from sklearn.ensemble import RandomForestRegressor

    >>> reg = RandomForestRegressor()
    >>> learner = jc.learners.SLearner(reg)
    >>> learner.fit(cf.np.X, cf.np.t, cf.np.y)



.. _Numpy: https://numpy.org/


