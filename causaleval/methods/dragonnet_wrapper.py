
from causaleval.methods.causal_method import CausalMethod
from causaleval.methods.dragonnet import dragonnet

class DragonNetWrapper(CausalMethod):

    def __init__(self, seed=0, learning_rate=0.001, num_epochs=5, num_covariates=25):
        super().__init__(seed)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_covariates = num_covariates

    def fit(self, x, t, y, refit=False) -> None:
        self.model = dragonnet.train_dragon(t, y, x)

    def predict_ite(self, x, t=None, y=None):
        # returns a 4-tupel for each instance : [y_0, y_1, t_pred, epislon]
        res = self.model.predict(x)
        return res[:, 1] - res[:, 0]

    def requires_provider(self):
        return False

    def __str__(self):
        return "DragonNet"





