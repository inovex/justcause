import numpy as np
import pandas as pd

from sacred import Experiment

from causaleval.methods.causal_method import CausalMethod
from causaleval.data.data_provider import DataProvider

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

    def prep_ite(self, data_provider, method):
        """

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :return:
        """
        x, t, y = data_provider.get_training_data()
        if method.requires_provider():
            method.fit_provider(data_provider)
        else:
            method.fit(x, t, y)
        return method.predict_ite(x)


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
        return np.sqrt(np.sum(np.power((1 - predicted/true), 2))/true.shape[0])

    @staticmethod
    def bias(true, predicted):
        return np.sum(predicted - true)/true.shape[0]

    def log_method(self, score_name, method, data_provider, score):
        self.ex.log_scalar(score_name + ',' + str(method) + ',' + str(data_provider), score)
        print(score_name + ',' + str(method) + ',' + str(data_provider)+ ',' + str(score))
        self.output = self.output.append(
            other={'metric': score_name, 'method': str(method), 'dataset': str(data_provider), 'score': score},
            ignore_index=True)

    def evaluate(self, data_provider, method):
        """

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :return:
        """

        # Setupt new DataFrame for every run of the metric
        self.output = pd.DataFrame(columns=['metric', 'method', 'dataset', 'score'])
        pred_ite = self.prep_ite(data_provider, method)
        true_ite = data_provider.get_true_ite()

        # TODO: Make a list of metric functions within this evaluation class and iterate over them
        self.log_method('PEHE', method, data_provider, self.pehe_score(true_ite, pred_ite))
        self.log_method('ATE', method, data_provider, self.ate_error(true_ite, pred_ite))
        self.log_method('ENORMSE', method, data_provider, self.enormse(true_ite, pred_ite))
        self.log_method('BIAS', method, data_provider, self.bias(true_ite, pred_ite))

class PlotEvaluation(EvaluationMetric):
    """Plot evaluation results of various metrics for further inspection
    Add the plots as artifacts to the Experiment.
    """

    def plot_bias_distribution(self):
        pass

    def plot_error_distribution(self):
        pass


