import numpy as np

from itertools import cycle


class DataProvider():

    def __init__(self):
        # Standard null initialization
        self.x = None
        self.y = None
        self.t = None
        self.y_cf = None
        self.y_0 = None
        self.y_1 = None
        self.train_selection = None
        self.test_selection = None

        self.load_training_data()
        self.set_train_test_split(train_size=0.8) # Use 80/20 as default

    def __str__(self):
        return "Abstract Data Provider"

    def load_training_data(self):
        """
        Overwrite this function to set x,y,t,y_cf with the required values for the specific dataset

        :return:
        """
        raise NotImplementedError('not yet implemented for ' + str(self))

    def get_training_data(self, size=None):
        if self.y is None:
            self.load_training_data()

        if size is None:
            return self.x[self.train_selection], self.t[self.train_selection], self.y[self.train_selection]
        else:
            if size > len(self.train_selection):
                raise AssertionError('requested training size too big for ' + str(self))

            # Choose a subset of the training_sample
            self.subselection = self.train_selection[np.random.choice(len(self.train_selection), size=size)]
            return self.x[self.subselection], self.t[self.subselection], self.y[self.subselection]

    def get_test_data(self):
        if self.y is None:
            self.load_training_data()

        return self.x[self.test_selection], self.t[self.test_selection], self.y[self.test_selection]

    def set_train_test_split(self, train_size=0.8):
        """Sets the train/test split for one evaluation run

        :param train_size: fraction of the whole data to be used as training
        """
        length = self.x.shape[0]
        self.train_selection = np.random.choice(length, size=int(train_size*length))
        full = np.arange(length)
        self.test_selection = full[~np.isin(full, self.train_selection)]

    def get_true_ite(self, data='train'):
        return self.y_1 - self.y_0

    def get_test_ite(self):
        return self.y_1[self.test_selection] - self.y_0[self.test_selection]

    def get_train_ite(self, subset=False):
        """

        :param subset: if true, return ite of the last retrieved subset (see get_training_data)
        :return:
        """
        if subset:
            return self.y_1[self.subselection] - self.y_0[self.subselection]
        else:
            return self.y_1[self.train_selection] - self.y_0[self.train_selection]

    def get_true_ate(self, subset=False):
        return np.mean(self.get_true_ite())

    def get_train_ate(self, subset=False):
        return np.mean(self.get_train_ite(subset))

    def get_test_ate(self, subset=False):
        return np.mean(self.get_train_ite(subset))

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
