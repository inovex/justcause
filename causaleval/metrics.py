import time
import os
import numpy as np
import pandas as pd

from sacred import Experiment

from causaleval.methods.causal_method import CausalMethod
from causaleval.data.data_provider import DataProvider

import config
import matplotlib
matplotlib.use(config.PLOT_BACKEND)
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt

import utils

class EvaluationMetric():

    def __init__(self, experiment, sizes=None, num_runs=1):
        """

        :param experiment: the sacred experiment in which this metric is called
        :type experiment: Experiment
        :param output: the sacred experiment in which this metric is called
        :type output: pd.DataFrame
        """
        self.ex = experiment
        self.output = None
        self.sizes = sizes
        self.num_runs = num_runs

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

    def __init__(self, experiment, sizes=None, num_runs=1, scores=['pehe', 'ate', 'bias', 'enormse']):
        super().__init__(experiment, sizes, num_runs)
        self.scores = scores


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

    @staticmethod
    def multi_run_function(true_ites, predicted_ites, function):
        """Evaluates a metric function with arguments true, predicted on the results of mutliple runs"""
        values = list(map(function, true_ites, predicted_ites))
        return np.mean(values)


    def log_all(self, method, data_provider, size, time_elapsed, pred_test, pred_train, true_test, true_train):

        function_map = {
            'PEHE' : self.pehe_score,
            'ATE' : self.ate_error,
            'ENORMSE' : self.enormse,
            'BIAS' : self.bias,

        }

        for key in function_map:
            self.log_method(key, method, data_provider, size, 'train', time_elapsed, function_map[key](pred_train, true_train))
            self.log_method(key, method, data_provider, size, 'test', time_elapsed, function_map[key](pred_test, true_test))

    def log_method(self, score_name, method, data_provider, size, sample, time, score):
        """Log output to console, csv and sacred logging

        :param score_name:
        :param method:
        :param data_provider:
        :param size:
        :param sample:
        :param score:
        """
        self.ex.log_scalar(score_name + ',' + str(method) + ',' + str(data_provider) + ',' + str(sample), round(score, 4))
        print(score_name + ',' + str(method) + ',' + str(data_provider)+ ',' + str(size) + ',' + str(sample) + ',' + str(time) + ',' + str(score))
        self.output = self.output.append(
            other={'metric': score_name, 'method': str(method), 'dataset': str(data_provider), 'size': size, 'sample': sample, 'time': time, 'score': score},
            ignore_index=True)

    def multi_run(self, method, data_provider, size, num_runs):
        train_true_ites = []
        test_true_ites = []
        train_predictions = []
        test_predictions = []


        function_map = {
            'PEHE-mean'+str(num_runs) : self.pehe_score,
            'ATE-mean'+str(num_runs): self.ate_error,
            'ENORMSE-mean'+str(num_runs) : self.enormse,
            'BIAS-mean'+str(num_runs) : self.bias,

        }

        start = time.time()

        for run in range(num_runs):
            # Perform evaluation for a number of runs
            pred_train, pred_test = self.prep_ite(data_provider, method, size=size)
            train_predictions.append(pred_train)
            test_predictions.append(pred_test)
            train_true_ites.append(data_provider.get_train_ite(subset=(size is not None)))
            test_true_ites.append(data_provider.get_test_ite())

        time_elapsed = time.time() - start

        # Work here with the accumulated ITE predictions for multi-run behaviour
        # e.g. log variance as a measure of robustness
        if config.SERVER and num_runs > 50:
            # write out files for plots, if number of runs is large on the server
            df_true = pd.DataFrame(data=train_true_ites)
            path = os.path.join(config.LOG_FILE_PATH, str(method)+'-'+str(data_provider)+'-'+str(num_runs)-'true')
            df_true.to_csv(path)
            path = os.path.join(config.LOG_FILE_PATH, str(method)+'-'+str(data_provider)+'-'+str(num_runs)-'train')
            df_pred = pd.DataFrame(data=train_true_ites)
            df_pred.to_csv(path)

        else:
            utils.robustness_plot(train_true_ites, train_predictions, str(method))
            utils.treatment_scatter(train_true_ites[0], train_predictions[0], str(method))
            utils.error_robustness_plot(list(map(self.pehe_score, train_true_ites, train_predictions)), str(method))


        if size is None:
            size = 'full'

        for key in function_map:
            if str(key).casefold().split("-")[0] in self.scores:
                # Only evaluate requested scores
                self.log_method(key, method, data_provider, size, 'train', time_elapsed,
                                self.multi_run_function(train_true_ites, train_predictions, function_map[key]))

                self.log_method(key, method, data_provider, size, 'test', time_elapsed,
                                self.multi_run_function(test_true_ites, test_predictions, function_map[key]))

        # Reset dataprovider
        data_provider.reset_cycle()

    def plot_residuals(self, pred_ite, true_ite):
        sns.distplot(true_ite, color='black')
        sns.distplot(pred_ite, color='gray')
        sns.distplot(true_ite - pred_ite, color='red')
        print(np.mean(true_ite))
        print(np.mean(pred_ite))
        plt.show()

    def evaluate(self, data_provider, method, plot=False):
        """

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :param sizes: The dataset sizes for which to evaluate
        :return:
        """

        # Setup new DataFrame for every run of the metric, then append later
        self.output = pd.DataFrame(columns=['metric', 'method', 'dataset', 'size', 'sample', 'time', 'score'])


        num_runs = self.num_runs

        if self.sizes:
            for size in self.sizes:
                # iterate over different sizes
                if num_runs > 1:
                    self.multi_run(method, data_provider, size, num_runs)

        else:
                self.multi_run(method, data_provider, size=None, num_runs=num_runs)


class PlotEvaluation(EvaluationMetric):
    """Plot evaluation results of various metrics for further inspection
    Add the plots as artifacts to the Experiment.
    """

    def plot_bias_distribution(self):
        pass

    def plot_error_distribution(self):
        pass


