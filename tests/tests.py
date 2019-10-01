from causaleval.data.generators.toy import SWagerDataProvider
from src.justcause.methods import SLearner
from causaleval.metrics import StandardEvaluation

from sklearn.linear_model import LinearRegression

import sacred
from sacred.observers import FileStorageObserver

from unittest import TestCase

def test_whole_experiment():
    """
    Currently, `sacred is very interwoven with the whole experiment process and thus unit-testing is hard.
    """

    ex = sacred.Experiment('normal')
    ex.observers.append(FileStorageObserver.create('results'))

    @ex.main
    def run(_run):
        method = SLearner(LinearRegression())
        data = SWagerDataProvider()
        metric = StandardEvaluation(ex, sizes=None, num_runs=1)
        metric.evaluate(data, method)
        assert len(metric.output.index) == 8 # 4 scores on train/test each
        assert metric.output['score'][0] != 0

        # test multirun
        metric = StandardEvaluation(ex, sizes=None, num_runs=5)
        metric.evaluate(data, method)


        # test varying sizes
        metric = StandardEvaluation(ex, sizes=[100, 200], num_runs=5)
        metric.evaluate(data, method)
        assert len(metric.output.index) == 16

    ex.run()

class IntegrationTests(TestCase):

    def setUp(self):
        self.ex = sacred.Experiment('normal')
        self.ex.observers.append(FileStorageObserver.create('results'))

    def tearDown(self):
        pass

    def test_experiment(self):
        """
        Tests the integration of method, data and metric into an experiment, logged via sacred

        requires a directory `results` for the FileStorageObserver

        :return:
        """
        test_whole_experiment()


    def test_dataprovider(self):

        data = SWagerDataProvider()
        x, t, y = data.get_training_data(size=500)
        self.assertEqual(len(t), len(y))
        self.assertEqual(len(y), len(x))
        self.assertEqual(len(t), 500)

        with self.assertRaises(AssertionError):
            data.get_training_data(size=2001)

        data.set_train_test_split(train_size=0.5)
        x, t, y = data.get_training_data(size=1000)
        self.assertEqual(len(t), 0.5*len(data.t))
        x_test, t_test, y_test = data.get_test_data()
        self.assertEqual(len(t_test), 1000)


    def test_rpy2(self):
        """
        Tests whether rpy2 is able to load the R environment and
        execute a causal forest
        """
        data = SWagerDataProvider()
        from src.justcause.methods import CausalForest
        cf = CausalForest()
        cf.fit(*data.get_training_data())
        self.assertIsNotNone(cf.predict_ite(*data.get_test_data()))





