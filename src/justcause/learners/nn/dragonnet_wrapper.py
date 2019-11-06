from justcause.contrib.dragonnet import dragonnet


class DragonNetWrapper:
    def __init__(
        self, learning_rate=0.001, num_epochs=50, num_covariates=25, batch_size=512,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_covariates = num_covariates

    def fit(self, x, t, y) -> None:
        self.model = dragonnet.train_dragon(
            t, y, x, num_epochs=self.num_epochs, batch_size=self.batch_size
        )

    def predict_ite(self, x, t=None, y=None):
        res = self.model.predict(x)
        return res[:, 1] - res[:, 0]

    def __str__(self):
        return "DragonNet" + str(self.num_epochs)
