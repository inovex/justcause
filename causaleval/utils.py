from sklearn.manifold import TSNE

import numpy as np
import config

# To make it work on MacOS
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("MacOSX")
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rcParams['axes.labelsize'] = 4
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['grid.color'] = 'black'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5

sns.set_style('whitegrid')

def get_regressor_name(representation):
    """

    :param representation: the string representation of a sklearn regressor
    :type representation: str
    :return:
    """
    if not isinstance(representation, str):
        representation = str(representation)

    return representation.split('(')[0]


# PLOTs

def surface_plot(y1, y0, y, y_cf, x):

    # Takes some time, thus use only smaller subsets
    covariates_2d = TSNE().fit_transform(x)

    fig = plt.figure(facecolor=(1,1,1))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y1, color=config.CYAN, alpha=1, s=2)
    ax.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y0, color=config.GREY, alpha=1, s=2)
    ax.view_init(10, 45)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y, color=config.BLUE, alpha=1, s=2)
    ax2.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y_cf, color=config.RED, alpha=1, s=2)
    ax2.view_init(10, 45)
    fig.tight_layout()
    plt.show()

def ite_plot(y1, y0):
    sns.distplot(y1 - y0, bins=100, color=config.BLUE)
    plt.show()

def plot_y_dist(y, y_cf):
    sns.distplot(y, bins=100, color=config.YELLOW)
    sns.distplot(y_cf, bins=100, color=config.RED)
    plt.show()

def simple_comparison_mean(y, t):
    treated = y[t==1]
    control = y[t==0]
    simple_mean = np.mean(treated) - np.mean(control)
    print('simple: ' + str(simple_mean))
