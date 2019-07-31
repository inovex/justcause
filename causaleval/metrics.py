import numpy as np

from sacred import Experiment

from causaleval.methods.causal_method import CausalMethod
from causaleval.data.data_provider import DataProvider

class EvaluationMetric():

    def __init__(self, experiment):
        """

        :param experiment: the sacred experiment in which this metric is called
        :type experiment: Experiment
        """
        self.ex = experiment

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
        method.fit(x, t, y)
        return method.predict_ite(x)


class StandardEvaluation(EvaluationMetric):
    """
    All the scores that work with full prediction result of the ITE on test data
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

    def bias(true, predicted):
        pass

    def evaluate(self, data_provider, method):
        """

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :return:
        """

        pred_ite = self.prep_ite(data_provider, method)
        true_ite = data_provider.get_true_ite()
        self.ex.log_scalar('PEHE, {method}, {dataset}'.format(method=str(method), dataset=str(data_provider)), self.pehe_score(true_ite, pred_ite))
        print('PEHE, {method}, {dataset}'.format(method=str(method), dataset=str(data_provider)), self.pehe_score(true_ite, pred_ite))
        self.ex.log_scalar('ATE, {method}, {dataset}'.format(method=str(method), dataset=str(data_provider)), self.ate_error(true_ite, pred_ite))
        print('ATE, {method}, {dataset}'.format(method=str(method), dataset=str(data_provider)), self.ate_error(true_ite, pred_ite))
        self.ex.log_scalar('ATE, {method}, {dataset}'.format(method=str(method), dataset=data_provider), self.enormse(true_ite, pred_ite))
        print('enormse score', self.enormse(true_ite, pred_ite))


