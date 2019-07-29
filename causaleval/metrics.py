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

    def evaluate(self, data_provider: type(DataProvider), method: type(CausalMethod)):
        """

        :param data_provider:
        :param method:
        :return:
        """
        pass


class EPEHE(EvaluationMetric):

    @staticmethod
    def pehe_score(true, predicted):
        return np.sqrt(np.mean(np.square(predicted - true)))

    def evaluate(self, data_provider, method):
        """

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :return:
        """

        x, t, y = data_provider.get_training_data()
        method.fit(*data_provider.get_training_data())
        pred_ite = method.predict_ite(x)
        score = self.pehe_score(data_provider.get_true_ite(), pred_ite)
        self.ex.log_scalar('pehe score', score)
        return score



    def evaluate_insample(self, data_provider, method):
        """

        :param data_provider:
        :type data_provider: DataProvider
        :param method:
        :type method: CausalMethod
        :return:
        """
