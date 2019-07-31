
from causaleval.methods.causal_method import CausalMethod
from causaleval.methods.ganite.ganite_model import GANITEModel

class GANITEWrapper(CausalMethod):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.model = GANITEModel(25, 0, num_treatments=2, nonlinearity="relu")

    


