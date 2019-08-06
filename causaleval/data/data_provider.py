import numpy as np

from itertools import cycle


class DataProvider():

    def __init__(self):
        # Standard null initialization
        self.x = None
        self.y = None
        self.t = None
        self.y_cf = None

    def __str__(self):
        return "Abstract Data Provider"

    def get_training_data(self, size=None):
        pass

    def get_test_data(self):
        pass

    def get_epoch_iterator(self, num_epochs):
        pass

    def get_true_ite(self, data=None):
        raise NotImplementedError('not yet implemented here')

    def get_true_ate(self, subset=None):
        np.mean(self.get_true_ite(subset))

    def get_train_generator_single(self, random=False, replacement=False):
        """
        Return a cycle generator of single objects
        :param random:
        :param replacement:
        :return:
        """
        if random:
            id_generator = cycle(np.random.choice(self.x.shape[0], size=self.x.shape[0], replacement=replacement))
        else:
            id_generator = cycle(range(self.x.shape[0]))
        while True:
            ID = next(id_generator)
            yield self.x[ID], self.t[ID], self.y[ID]


    def get_train_generator_batch(self, batch_size=32):
        num_batches = int(self.x.shape[0] / batch_size)
        batch_id_generator = cycle(range(num_batches))

        while True:
            id = next(batch_id_generator)
            start = id * batch_size
            end = (id + 1) * batch_size
            yield (self.x[start:end], self.t[start:end]), self.y[start:end]

    def get_num_covariates(self):
        return self.x.shape[1]

    def get_info(self):
        pass
