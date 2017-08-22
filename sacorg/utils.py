from math import factorial

def factorial(n):
    """
    Computes factorial of n
    :param n: An integer greater than zero
    :return factorial of n:
    """
    if n < 0:
        raise ValueError('The number must be non-negative')

    if n > 1:
        return n*factorial(n-1)
    else:
        return 1

def binomial(n, r):
    """
    Computes binomial coefficient (n,r)
    :param n:
    :param r:
    :return:
    """
    val = factorial(n) / (factorial(r) * factorial(n - r))
    return val
