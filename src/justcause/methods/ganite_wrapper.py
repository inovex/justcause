
from .causal_method import CausalMethod
from .data.data_provider import DataProvider
from .ganite.ganite_model import GANITEModel
import numpy as np

class GANITEWrapper(CausalMethod):
    def __init__(self, seed=0, learning_rate=0.0001, num_epochs=50, num_covariates=25, verbose=False):
        super().__init__(seed)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_covariates = num_covariates
        self.verbose = verbose
        self.model = self.build_model(num_covariates)

    def fit(self, x, t, y, refit=False) -> None:
        raise NotImplementedError( str(self) + ' requires a batch generator for fitting')

    def fit_provider(self, data_provider):
        """
        :param data_provider:
        :type data_provider: DataProvider
        """
        # See if the model suits the data
        if not self.does_fit_provider(data_provider):
            self.model = self.build_model(data_provider.get_num_covariates())

        batch_gen = data_provider.get_train_generator_batch(batch_size=128)
        self.model.train(train_generator=batch_gen,
                         train_steps=10,
                         val_generator=batch_gen,
                         val_steps=10,
                         num_epochs=self.num_epochs,
                         learning_rate=0.0001, verbose=self.verbose)

    def predict_ite(self, x, t=None, y=None):
        ys_cf = self.model.pred_full(x, t, y)
        return ys_cf[:, 1] - ys_cf[:, 0]

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


    def build_model(self, num_covariates):
        """
        Rebuild model with new input dimension
        :param num_covariates: number of covariates that are used as input to the model
        """
        self.model = GANITEModel(num_covariates, None, num_treatments=2)
        return self.model









