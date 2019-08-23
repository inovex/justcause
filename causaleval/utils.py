from sklearn.manifold import TSNE

import numpy as np
import config

# To make it work on MacOS
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use(config.PLOT_BACKEND)
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rcParams['axes.labelsize'] = 4
plt.rcParams['figure.facecolor'] = '1'
plt.rcParams['grid.color'] = 'black'
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams["savefig.dpi"] = 300

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

def surface_plot(y1, y0, y, y_cf, x, name='default'):

    # Takes some time, thus use only smaller subsets
    covariates_2d = TSNE().fit_transform(x)

    fig = plt.figure(facecolor=(1,1,1))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y1, color=config.CYAN, alpha=1, s=2, label='Treated')
    ax.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y0, color=config.GREY, alpha=1, s=2, label='Control')
    ax.view_init(10, 45)
    ax.set_zlabel('Outcome', fontsize=10)
    ax.legend()
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y, color=config.BLUE, alpha=1, s=2, label='Observed')
    ax2.scatter(covariates_2d[:, 0], covariates_2d[:, 1], y_cf, color=config.RED, alpha=1, s=2, label='Counterfactual')
    ax2.view_init(10, 45)
    ax2.set_zlabel('Outcome', fontsize=10)
    ax2.legend()
    fig.tight_layout()
    if config.PLOT_WRITE:
        path = config.RESULT_PLOT_PATH
        plt.savefig(path + '/' + name + '-surface-plot')
        plt.clf()
    else:
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

def robustness_plot(true_ites, predicted_ites, method_name='Method'):
    true_ates = np.mean(true_ites, axis=1)
    predicted_ates = np.mean(predicted_ites, axis=1)
    sns.set_style('whitegrid')
    sns.lineplot(x=np.arange(len(true_ates)), y=true_ates, color=config.BLUE, label='True')
    sns.lineplot(x=np.arange(len(true_ates)), y=predicted_ates, color=config.RED, label='Estimated')
    plt.xlabel('#runs')
    plt.ylabel('average treatment effect')
    plt.title(method_name)
    plt.legend()
    if config.PLOT_WRITE:
        path = config.RESULT_PLOT_PATH
        plt.savefig(path + '/' + method_name + 'robustness')
        plt.clf()
    else:
        plt.show()

def treatment_scatter(true_ite, predicted_ite, method_name='Method'):
    sns.set_style('whitegrid')
    sns.scatterplot(true_ite, predicted_ite, color=config.GREY)
    plt.xlabel('true effect')
    plt.ylabel('predicted effect')
    plt.title(method_name)
    if config.PLOT_WRITE:
        path = config.RESULT_PLOT_PATH
        plt.savefig(path + '/' + method_name + 'treatment-scatter')
        plt.clf()
    else:
        plt.show()

def error_robustness_plot(errors, method_name='Method'):
    sns.set_style('whitegrid')
    sns.lineplot(x=np.arange(len(errors)), y=errors, color=config.RED, label='PEHE')
    plt.xlabel('#runs')
    plt.ylabel('error score')
    plt.title(method_name)
    plt.legend()
    if config.PLOT_WRITE:
        path = config.RESULT_PLOT_PATH
        plt.savefig(path + '/' + method_name + 'error-robustness')
        plt.clf()
    else:
        plt.show()

