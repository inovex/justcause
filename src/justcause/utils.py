from numpy.random import RandomState

MAX_INT = 2 ** 32 - 1


def int_from_random_state(random_state: RandomState):
    """Samples an integer from a RandomState to be passed on as seed"""
    if isinstance(random_state, int):
        return random_state
    if isinstance(random_state, RandomState):
        return int(random_state.randint(0, MAX_INT, size=1))
    else:
        raise ValueError("Only RandomState and int can be parsed to int")
