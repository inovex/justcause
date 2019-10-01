from ..data_provider import DataProvider


class DataGenerator(DataProvider):


    def __init__(self, params):

        # Make the parameters to instance attributes
        for key in params:
            setattr(self, key, params[key])

        self.params = params

        # Call super constructor only after setting the attributes to make sure all the init functions work
        super().__init__()

