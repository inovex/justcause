def replace_factual_outcomes(y_0, y_1, y, t):
    """ Replaces the predicted components with factual observations where possible

    Args:
        y_0: predicted control outcomes
        y_1: predicted treatment outcomes
        y: factual outcomes
        t: factual treatment indicators

    Returns: y_0, y_1 with factual outcomes replaced where possible
    """
    for i in range(len(t)):
        if t[i] == 1:
            y_1 = y[i]
        else:
            y_0 = y[i]
    return y_0, y_1
