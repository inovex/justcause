import numpy as np


# ToDo: Check if this is the smartest thing to do it like that
def get_regressor_name(representation):
    """

    :param representation: the string representation of a sklearn regressor
    :type representation: str
    :return:
    """
    if not isinstance(representation, str):
        representation = str(representation)

    return representation.split("(")[0]


def simple_comparison_mean(y, t):
    treated = y[t == 1]
    control = y[t == 0]
    simple_mean = np.mean(treated) - np.mean(control)
    print("simple: " + str(simple_mean))
