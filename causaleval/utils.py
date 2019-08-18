from sklearn.manifold import TSNE

import numpy as np

# To make it work on MacOS
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("MacOSX")
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt

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

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y1, color='green', alpha=0.7, s=2)
    ax.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y0, color='gray', alpha=0.7, s=2)
    ax.view_init(30, 45)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y, color='blue', alpha=0.7, s=2)
    ax2.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y_cf, color='red', alpha=0.5, s=2)
    ax2.view_init(30, 45)
    plt.show()

def ite_plot(y1, y0):
    sns.distplot(y1 - y0, bins=100, color='blue')
    plt.show()

def plot_y_dist(y, y_cf):
    sns.distplot(y, bins=100, color='blue')
    sns.distplot(y_cf, bins=100, color='red')
    plt.show()

def simple_comparison_mean(y, t):
    treated = y[t==1]
    control = y[t==0]
    simple_mean = np.mean(treated) - np.mean(control)
    print('simple: ' + str(simple_mean))
