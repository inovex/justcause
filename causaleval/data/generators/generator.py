from causaleval.data.data_provider import DataProvider

class DataGenerator(DataProvider):


    def __init__(self, params):
        super().__init__()
        self.params = params
        # Make the parameters to instance attributes
        for key in params:
            setattr(self, key, params[key])

        self.params = params

