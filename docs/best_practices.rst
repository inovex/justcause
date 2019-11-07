==============
Best Practices
==============

Dependency Management & Reproducibility
=======================================

1. Always keep your abstract (unpinned) dependencies updated in ``environment.yaml`` and eventually
   in ``setup.cfg`` if you want to ship and install your package via ``pip`` later on.
2. Create concrete dependencies as ``environment.lock.yaml`` for the exact reproduction of your
   environment with::

    conda env export -n justcause -f environment.lock.yaml

   For multi-OS development, consider using ``--no-builds`` during the export.
3. Update your current environment with respect to a new ``environment.lock.yaml`` using::

    conda env update -f environment.lock.yaml --prune



Organization, Logging and Reproduction of Experiments
=====================================================

(WIP) Mention here Sacred and explain how to use it.