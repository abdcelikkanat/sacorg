from sacorg.algorithms.mcmc import mcmc
from sacorg.algorithms.blitzstein_diaconis import blitzstein_diaconis
from sacorg.algorithms.myalg import myalg


def count(deg_seq, method, num_of_samples=1000, verbose=False):

    if method == "BD":
        estimate, std = blitzstein_diaconis.count(deg_seq=deg_seq, num_of_samples=num_of_samples, verbose=verbose)
        return estimate, std

    if method == "C":
        result = myalg.count(deg_seq=deg_seq, verbose=verbose)
        return result


def get_sample(deg_seq, method, num_of_samples, iteration=-1, verbose=False):

    if method == "MCMC":
        return mcmc.get_sample(deg_seq=deg_seq, num_of_samples=num_of_samples, iteration=iteration, verbose=verbose)

    if method == "BD":
        return blitzstein_diaconis.get_sample(deg_seq=deg_seq, num_of_samples=num_of_samples, verbose=verbose)

    if method == "C":
        return myalg.get_sample(deg_seq=deg_seq, num_of_samples=num_of_samples, verbose=verbose)
