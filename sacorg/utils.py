from math import factorial

def binomial(n, r):
    """
    Computes binomial coefficient (n,r)
    :param n:
    :param r:
    :return:
    """
    val = factorial(n) / (factorial(r) * factorial(n - r))
    return val
