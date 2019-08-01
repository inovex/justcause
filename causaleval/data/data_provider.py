
class DataProvider():

    def __init__(self):
        pass

    def __str__(self):
        return "Abstract Data Provider"

    def get_training_data(self, size=None):
        pass

    def get_test_data(self):
        pass

    def get_epoch_iterator(self, num_epochs):
        pass

    def get_true_ite(self, data=None):
        pass

    def get_true_ate(self, subset=None):
        pass

    def get_train_generator_batch(self, batch_size=32):
        raise NotImplementedError('Batch generator not implemented for ', str(self))

    def get_train_generator_single(self, random=False, replacement=False):
        raise NotImplementedError('Instance generator not implemented for ', str(self))

    def get_num_covariates(self):
        raise NotImplementedError('Not implemented for ', str(self))

    def get_info(self):
        pass
