import os

import numpy as np

from .tools import data_from_generative_function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def multi_modal_effect(X):
    prob = sigmoid(X[:, 7]) > 0.5
    return np.random.normal((3 * prob) + 1 * (1 - prob), 0.1, size=len(X))


def example_emcs():
    path = "/Users/MaximilianFranz/Documents/ba/eval/justcause/datasets/ihdp-cfr"
    fname = os.path.join(path, "train.npz")
    train = np.load(fname)

    covariates = train["x"][:, :, 0]  # has shape (672, 25, 100)

    def treatment_assignment(cov, size=None):
        return np.random.binomial(1, p=sigmoid(cov[:, 7]), size=size)

    def outcome_assignment(cov, size=None):
        Y_0 = np.random.normal(cov[:, 0] + 1, 0.2, size=size)
        Y_1 = Y_0 + multi_modal_effect(cov)
        return Y_0, Y_1

    return data_from_generative_function(
        covariates, treatment_assignment, outcome_assignment
    )
