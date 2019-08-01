
from causaleval.methods.causal_method import CausalMethod
from causaleval.data.data_provider import DataProvider
from causaleval.methods.ganite.ganite_model import GANITEModel

class GANITEWrapper(CausalMethod):
    def __init__(self, seed=0, learning_rate=0.001, num_epochs=5, num_covariates=25):
        super().__init__(seed)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_covariates = num_covariates
        self.model = self.build_model(num_covariates)

    def fit(self, x, t, y, refit=False) -> None:
        raise NotImplementedError('GANITE requires a batch generator for fitting')

    def fit_provider(self, data_provider):
        """
        :param data_provider:
        :type data_provider: DataProvider
        """
        # See if the model suits the data
        if not self.does_fit_provider(data_provider):
            self.rebuild_model(data_provider.get_num_covariates())

        batch_gen = data_provider.get_train_generator_batch(batch_size=64)
        self.model.train(batch_gen, 5, batch_gen, 5, 50, 0.001)

    def predict_ite(self, x):
        ys = self.model.predict(x)
        return ys[:, 1] - ys[:, 0]

    def requires_provider(self):
        return True

    def __str__(self):
        return "GANITE"

    def does_fit_provider(self, data_provider):
        """

        :param data_provider: DataProvider
        :return:
        """
        return self.num_covariates is data_provider.get_num_covariates()


    def build_model(self, num_covariates, ):
        """
        Rebuild model with new input dimension
        :param num_covariates: number of covariates that are used as input to the model
        """
        self.model = GANITEModel(num_covariates, None, num_treatments=2)
        return self.model









