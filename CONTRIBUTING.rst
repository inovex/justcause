============
Contributing
============


Issue Reports
=============

If you experience bugs or in general issues with JustCause, please file an
issue report on our `issue tracker`_.


Code Contributions
==================

Submit an issue
---------------

Before you work on any non-trivial code contribution it's best to first create
an issue report to start a discussion on the subject. This often provides
additional considerations and avoids unnecessary work.


Clone the repository
--------------------

#. `Create a Gitub account`_  if you do not already have one.
#. Fork the `project repository`_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on the GitHub server.
#. Clone this copy to your local disk::

    git clone git@github.com:YourLogin/justcause.git

#. Create an environment ``justcause`` with the help of `Miniconda`_ and activate it::

    conda env create -f environment.yaml
    conda activate justcause

#. Install ``justcause`` with::

    python setup.py develop

#. Install ``pre-commit``::

    pip install pre-commit
    pre-commit install

   JustCause comes with a lot of hooks configured to
   automatically help you with providing clean code.

#. Create a branch to hold your changes::

    git checkout -b my-feature

   and start making changes. Never work on the master branch!

#. Start your work on this branch. When you’re done editing, do::

    git add modified_files
    git commit

   to record your changes in Git, then push them to GitHub with::

    git push -u origin my-feature

#. Please check that your changes don't break any unit tests with::

    py.test

   Don't forget to also add unit tests in case your contribution
   adds an additional feature and is not just a bugfix.

#. Add yourself to the list of contributors in ``AUTHORS.rst``.
#. Go to the web page of your JustCause fork, and click
   "Create pull request" to send your changes to the maintainers for review.
   Find more detailed information `creating a PR`_.

Release
=======

As a JustCause maintainer following steps are needed to release a new version:

#. Make sure all unit tests on `Cirrus-CI`_ are green.
#. Update the ``CHANGELOG.rst`` file.
#. Tag the current commit on the master branch with a release tag, e.g. ``v1.2.3``.
#. Clean up the ``dist`` and ``build`` folders with ``rm -rf dist build``
   to avoid confusion with old builds and Sphinx docs.
#. Run ``python setup.py dists`` and check that the files in ``dist`` have
   the correct version (no ``.dirty`` or Git hash) according to the Git tag.
   Also sizes of the distributions should be less than 500KB (for bdist), otherwise unwanted
   clutter may have been included.
#. Make sure you uploaded the new tag to Github, run ``git push origin --tags``.
#. Run ``twine upload dist/*`` and check that everything was uploaded to `PyPI`_ correctly.


.. _Cirrus-CI: https://cirrus-ci.com/github/inovex/justcase
.. _PyPI: https://pypi.python.org/
.. _project repository: https://github.com/inovex/justcause/
.. _Git: http://git-scm.com/
.. _Miniconda: https://conda.io/miniconda.html
.. _issue tracker: http://github.com/inovex/justcause/issues
.. _Create a Gitub account: https://github.com/signup/free
.. _creating a PR: https://help.github.com/articles/creating-a-pull-request/
.. _tox: https://tox.readthedocs.io/
.. _flake8: http://flake8.pycqa.org/
