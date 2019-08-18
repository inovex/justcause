import time
import numpy as np
import pandas as pd

from sacred import Experiment

from causaleval.methods.causal_method import CausalMethod
from causaleval.data.data_provider import DataProvider

import matplotlib
matplotlib.use("MacOSX")
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt


class EvaluationMetric():

    def __init__(self, experiment):
        """

        :param experiment: the sacred experiment in which this metric is called
        :type experiment: Experiment
        :param output: the sacred experiment in which this metric is called
        :type output: pd.DataFrame
        """
        self.ex = experiment
        self.output = None

    def evaluate(self, data_provider, method):
        """

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :return:
        """
        pass

    def prep_ite(self, data_provider, method, size=None):
        """ Return the Predicted ITE on train and test data

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :param size: for which to evaluate
        :return:
        """
        x, t, y = data_provider.get_training_data(size=size)
        if method.requires_provider():
            method.fit_provider(data_provider)
        else:
            method.fit(x, t, y)

        x_test, t_test, y_test = data_provider.get_test_data()

        return method.predict_ite(x, t, y), method.predict_ite(x_test, t_test, y_test)


class StandardEvaluation(EvaluationMetric):
    """All the scores that work with full prediction result of the ITE on test data
    """

    @staticmethod
    def pehe_score(true, predicted):
        return np.sqrt(np.mean(np.square(predicted - true)))

    @staticmethod
    def ate_error(true, predicted):
        return np.abs(np.mean(true) - np.mean(predicted))

    @staticmethod
    def enormse(true, predicted):
        return np.sqrt(np.sum(np.power((1 - (predicted + 0.0001) /(true + 0.0001)), 2))/true.shape[0])

    @staticmethod
    def bias(true, predicted):
        return np.sum(predicted - true)/true.shape[0]

    def log_method(self, score_name, method, data_provider, size, sample, time, score):
        """Log output to console, csv and sacred logging

        :param score_name:
        :param method:
        :param data_provider:
        :param size:
        :param sample:
        :param score:
        """
        self.ex.log_scalar(score_name + ',' + str(method) + ',' + str(data_provider) + ',' + str(sample), score)
        print(score_name + ',' + str(method) + ',' + str(data_provider)+ ',' + str(size) + ',' + str(sample) + ',' + str(time) + ',' + str(score))
        self.output = self.output.append(
            other={'metric': score_name, 'method': str(method), 'dataset': str(data_provider), 'size': size, 'sample': sample, 'time': time, 'score': score},
            ignore_index=True)

    def plot_residuals(self, pred_ite, true_ite):
        sns.distplot(true_ite, color='green')
        sns.distplot(pred_ite, color='gray')
        print(np.mean(true_ite))
        print(np.mean(pred_ite))
        plt.show()

    def evaluate(self, data_provider, method, sizes=None, plot=False):
        """

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :param sizes: The dataset sizes for which to evaluate
        :return:
        """

        function_map = {
            'PEHE' : self.pehe_score,
            'ATE' : self.ate_error,
            'ENORMSE' : self.enormse,
            'BIAS' : self.bias,

        }

        self.output = pd.DataFrame(columns=['metric', 'method', 'dataset', 'size', 'sample', 'time', 'score'])

        # Setupt new DataFrame for every run of the metric

        if sizes:
            for size in sizes:
                start = time.time()
                pred_train, pred_test = self.prep_ite(data_provider, method, size=size)
                true_train = data_provider.get_train_ite(subset=True)
                true_test = data_provider.get_test_ite()
                time_elapsed = round(time.time() - start, 3)

                for key in function_map:
                    self.log_method(key, method, data_provider, size, 'train', time_elapsed, function_map[key](pred_train, true_train))
                    self.log_method(key, method, data_provider, size, 'test', time_elapsed, function_map[key](pred_test, true_test))
        else:
            start = time.time()
            pred_train, pred_test = self.prep_ite(data_provider, method, size=None)
            true_train = data_provider.get_train_ite(subset=False)
            true_test = data_provider.get_test_ite()
            time_elapsed = round(time.time() - start, 3)

            for key in function_map:
                self.log_method(key, method, data_provider, 'full', 'test', time_elapsed, function_map[key](pred_test, true_test))
                self.log_method(key, method, data_provider, 'full', 'train', time_elapsed, function_map[key](pred_train, true_train))

class PlotEvaluation(EvaluationMetric):
    """Plot evaluation results of various metrics for further inspection
    Add the plots as artifacts to the Experiment.
    """

    def plot_bias_distribution(self):
        pass

    def plot_error_distribution(self):
        pass


