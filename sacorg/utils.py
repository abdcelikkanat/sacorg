from math import factorial

def binomial(n, r):
    """
    Computes binomial coefficient (n,r)
    :param n:
    :param r:
    :return:
    """
    try:
        val = factorial(n) / (factorial(r) * factorial(n - r))
    except ValueError:
        val = 0

    return val