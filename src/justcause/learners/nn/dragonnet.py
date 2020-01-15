import numpy as np

from ..utils import replace_factual_outcomes


class DragonNet:
    """Wrapper of the DragonNet implementation in `justcause.contrib.dragonnet`

     Original code taken with slide adaption for usage within the framework.
     Source can be found here: https://github.com/claudiashi57/dragonnet

     References:
         [1] C. Shi, D. M. Blei, and V. Veitch,
            “Adapting Neural Networks for the Estimation of Treatment Effects.”
     """

    def __init__(
        self, learning_rate=0.001, num_epochs=50, batch_size=512, validation_split=0.1
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "DragonNet(epochs={}, lr={}, batch={}, val_split={})".format(
            self.num_epochs, self.learning_rate, self.batch_size, self.validation_split
        )

    def fit(self, x: np.array, t: np.array, y: np.array) -> None:
        """Trains DragonNet with hyper-parameters specified in the constructor

        Args:
            x: covariates for all instances, shape (num_instance, num_features)
            t: treatment indicator vector, shape (num_instance)
            y: factual outcomes, shape (num_instance)

        """
        # Late import to avoid installing all of dragonnet's requirements
        from ...contrib.dragonnet import dragonnet

        self.model = dragonnet.train_dragon(
            t,
            y,
            x,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            val_split=self.validation_split,
        )

    def predict_ite(
        self,
        x: np.array,
        t: np.array = None,
        y: np.array = None,
        return_components: bool = False,
        replace_factuals: bool = False,
    ):
        """Predicts ITE for the given samples

        Args:
            x: covariates in shape (num_instances, num_features)
            t: treatment indicator, binary in shape (num_instances)
            y: factual outcomes in shape (num_instances)
            return_components: whether to return Y(0) and Y(1) predictions separately
            replace_factuals: whether to replace outcomes with true outcomes
                where possible

        Returns:
            a vector of ITEs for the inputs; also returns Y(0) and Y(1) for all
            inputs if ``return_components`` is ``True``
        """
        assert self.model is not None, "DragonNet must be fit before use"

        res = self.model.predict(x)
        y_0, y_1 = res[:, 0], res[:, 1]

        if return_components:
            if t is not None and y is not None and replace_factuals:
                y_0, y_1 = replace_factual_outcomes(y_0, y_1, y, t)
            return y_1 - y_0, y_0, y_1
        else:
            return y_1 - y_0

    def estimate_ate(
        self, x: np.array, t: np.array = None, y: np.array = None,
    ) -> float:
        """Estimates the average treatment effect of the given population

        First, it fits the model on the given population, then predicts ITEs and uses
        the mean as an estimate for the ATE

        Args:
            x: covariates in shape (num_instances, num_features)
            t: treatment indicator, binary in shape (num_instances)
            y: factual outcomes in shape (num_instances)

        Returns: ATE estimate as the mean of ITEs
        """
        self.fit(x, t, y)
        ite = self.predict_ite(x, t, y)
        return float(np.mean(ite))
