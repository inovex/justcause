=========
Changelog
=========

Version 0.4
===========

- removed dependency to ``causalml`` thus x- and t- learner were removed
- removed dependency to ``rpy`` thus CausalForest method was removed
- removed dependency to ``tensorflow`` thus ganite and dragonnet were removed
- added missing ``requests`` library in dependencies

Version 0.3.2
=============
- some fixes in usage documentation
- fixed persistent "future annotation"-bug that prevent importing justcause on Python 3.6
- ``generate_data`` handles now CausaFrames as covariates correctly

Version 0.3.1
=============
- bugfixes in data generator based on IHDP (wrong covariate was used)
- bugfix in evaluation (train/test results were in fact the same)
- Support for lower Python versions (type annotations as used in JustCause were only available for Python >=3.7)

Version 0.3
===========

- data sets and generators now return a list of CausalFrames instead of iterators
- treatment column is always ``t`` and outcome column always ``y``
- improved and extended documentation

Version 0.2
===========

- Complete overhaul of everything that was done before in order to have:

  - a distributable Python package
  - unit tests
  - a consistent API
  - some documentation
  - and much more ;-)


Version 0.1
===========

- Reflecting the state of the finished bachelor thesis
