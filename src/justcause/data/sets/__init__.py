"""Contains some of the most used reference data sets for method evaluation

For a more detailed explanation of the data sets, see Section 4.2 "Existing
Benchmarking Datasets" in the thesis under (docs/thesis-mfranz.pdf) [1]

BEWARE: The performance of the data set accessors is highly dependent on how
they are called. For example:

    reps = load_ihdp()[0:100]

is much slower than:

    reps = load_ihdp(select_rep=np.arange(100))

because in the first case, all data is first fetch and created, while the second call
only merges the outcomes and covariates for the replications requested.

References:
    [1] Maximilian Franz, "A Systematic Review of Machine Learning Estimators for
    Causal Effects", Bachelor Thesis, Karlsruhe Institute of Technology, 2019.
    See `docs/thesis-mfranz.pdf`.

"""
from .ibm import load_ibm
from .ihdp import load_ihdp
from .twins import load_twins

__all__ = ["load_ibm", "load_ihdp", "load_twins"]
