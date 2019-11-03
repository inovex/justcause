"""Plotting utilities for the examples in the notebook

ToDo: Rework these
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

plt.style.use("seaborn")
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.facecolor"] = "1"
plt.rcParams["grid.color"] = "black"
plt.rcParams["grid.linestyle"] = ":"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["savefig.dpi"] = 300

sns.set_style("whitegrid")

NUM_BINS = 200
# Todo: Check the definition of seaborn's colors and aliases like `red`, `blue`
CYAN = "#4ECDC4"
BLUE = "#59D2FE"
RED = "#FF6B6B"
YELLOW = "#FAA916"
GREY = "#4A6670"
COLOR_LIST = [CYAN, BLUE, RED, YELLOW, GREY]
# ToDo: get rid of these variables below and rather pass them as parameters
PLOT_WRITE = True
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(ROOT_DIR, "results") + "/"
RESULT_PLOT_PATH = os.path.join(OUTPUT_PATH, "plots")


def surface_plot(y1, y0, y, y_cf, x, name="default"):
    # Takes some time, thus use only smaller subsets
    sns.set_style("whitegrid")
    covariates_2d = TSNE().fit_transform(x)

    fig = plt.figure(facecolor=(1, 1, 1))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(
        covariates_2d[:, 0],
        covariates_2d[:, 1],
        y1,
        color=CYAN,
        alpha=1,
        s=2,
        label="Treated",
    )
    ax.scatter(
        covariates_2d[:, 0],
        covariates_2d[:, 1],
        y0,
        color=GREY,
        alpha=1,
        s=2,
        label="Control",
    )
    ax.view_init(10, 45)
    ax.set_zlabel("Outcome", fontsize=10)
    ax.legend()
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(
        covariates_2d[:, 0],
        covariates_2d[:, 1],
        y,
        color=BLUE,
        alpha=1,
        s=2,
        label="Observed",
    )
    ax2.scatter(
        covariates_2d[:, 0],
        covariates_2d[:, 1],
        y_cf,
        color=RED,
        alpha=1,
        s=2,
        label="Counterfactual",
    )
    ax2.view_init(10, 45)
    ax2.set_zlabel("Outcome", fontsize=10)
    ax2.legend()
    fig.tight_layout()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + name + "-surface-plot")
        plt.clf()
    else:
        plt.show()


def ite_plot(y1, y0, write=False, method_name="default"):
    sns.set_style("whitegrid")
    sns.distplot(y1 - y0, bins=NUM_BINS, color=BLUE)
    plt.title(method_name)
    plt.tight_layout()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + method_name + "-ite")
        plt.clf()
    else:
        plt.show()


def plot_y_dist(y, y_cf, write=False, method_name="default"):
    sns.set_style("whitegrid")
    sns.distplot(y, bins=NUM_BINS, color=BLUE, label="Observed")
    sns.distplot(y_cf, bins=NUM_BINS, color=RED, label="Counterfactual")
    plt.xlabel("Outcomes")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + method_name + "-ydist")
        plt.clf()
    else:
        plt.show()


def dist_plot(x, method_name="default"):
    sns.set_style("whitegrid")
    sns.distplot(x, bins=NUM_BINS, color=BLUE)
    plt.tight_layout()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + method_name + "-ydist")
        plt.clf()
    else:
        plt.show()


def true_ate_plot(true_ates, dataset="default"):
    sns.set_style("whitegrid")
    sns.lineplot(
        x=np.arange(len(true_ates)),
        y=true_ates,
        color=BLUE,
        label="True Average Treatment Effect",
    )
    plt.xlabel("Replications")
    plt.ylabel("Average Treatment Effect")
    plt.title(dataset)
    plt.legend()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + dataset + "true-ate")
        plt.clf()
    else:
        plt.show()


def confounder_outcome_plot(confounder, y_1, dataset="default"):
    sns.set_style("whitegrid")
    sns.scatterplot(x=confounder, y=y_1, color=BLUE)
    plt.xlabel("Confounder")
    plt.ylabel("Treatment Effect")
    plt.title(dataset)
    plt.legend()
    plt.tight_layout()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + dataset + "confounder-outcome")
        plt.clf()
    else:
        plt.show()


def line_plot(x, y, label="default", xlabel="xlabel", ylabel="ylabel"):
    sns.set_style("whitegrid")
    sns.lineplot(x=x, y=y, color=BLUE, label="label")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + label + "line_plot")
        plt.clf()
    else:
        plt.show()


def scatter_plot(x, y, label="default", xlabel="xlabel", ylabel="ylabel"):
    sns.set_style("whitegrid")
    sns.scatterplot(x=x, y=y, color=BLUE, label="label")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + label + "line_plot")
        plt.clf()
    else:
        plt.show()


def true_ate_dist_plot(true_ates, dataset="default"):
    sns.set_style("whitegrid")
    sns.distplot(true_ates, color=BLUE, label="Density")
    plt.title(dataset)
    plt.legend()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + dataset + "true-ate-dist")
        plt.clf()
    else:
        plt.show()


def propensity_distribution_plot(data_provider):
    sns.set_style("whitegrid")
    reg = LogisticRegression()
    reg.fit(data_provider.x, data_provider.t)
    propensity = reg.predict_proba(data_provider.x)[:, 1]
    sns.distplot(propensity, color=BLUE, bins=NUM_BINS)
    plt.xlabel("Probability of Treatment")
    plt.ylabel("Density")
    plt.title("Propensity Distribution")
    plt.legend()
    plt.tight_layout()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + str(data_provider) + "propensity")
        plt.clf()
    else:
        plt.show()


def robustness_plot(true_ites, predicted_ites, method_name="Method"):
    true_ates = np.mean(true_ites, axis=1)
    predicted_ates = np.mean(predicted_ites, axis=1)
    sns.set_style("whitegrid")
    sns.lineplot(x=np.arange(len(true_ates)), y=true_ates, color=BLUE, label="True")
    sns.lineplot(
        x=np.arange(len(true_ates)), y=predicted_ates, color=RED, label="Estimated",
    )
    plt.xlabel("Replications")
    plt.ylabel("Average Treatment Effect")
    plt.title(method_name)
    plt.legend()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + method_name + "robustness")
        plt.clf()
    else:
        plt.show()


def treatment_scatter(true_ite, predicted_ite, method_name="Method"):
    sns.set_style("whitegrid")
    sns.scatterplot(true_ite, predicted_ite, color=GREY)
    plt.xlabel("true effect")
    plt.ylabel("predicted effect")
    plt.title(method_name)
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + method_name + "treatment-scatter")
        plt.clf()
    else:
        plt.show()


def error_robustness_plot(errors, method_name="Method"):
    sns.set_style("whitegrid")
    sns.lineplot(x=np.arange(len(errors)), y=errors, color=RED, label="PEHE")
    plt.xlabel("#runs")
    plt.ylabel("error score")
    plt.title(method_name)
    plt.legend()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + method_name + "error-robustness")
        plt.clf()
    else:
        plt.show()


def error_distribution_plot(errors, method_name="Method"):
    sns.set_style("whitegrid")
    sns.distplot(errors, bins=NUM_BINS, color=BLUE)  # Plot distribution
    plt.axvline(np.median(errors), 0, 1, color=RED)  # Plot mean as vertical line
    plt.axvline(np.mean(errors), 0, 1, color=YELLOW)  # Plot mean as vertical line
    plt.title(method_name)
    plt.legend()
    if PLOT_WRITE:
        path = RESULT_PLOT_PATH
        plt.savefig(path + "/" + method_name + "error-distribution")
        plt.clf()
    else:
        plt.show()
