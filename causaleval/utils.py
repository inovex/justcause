
def get_regressor_name(representation):
    """

    :param representation: the string representation of a sklearn regressor
    :type representation: str
    :return:
    """
    if not isinstance(representation, str):
        representation = str(representation)

    return representation.split('(')[0]
