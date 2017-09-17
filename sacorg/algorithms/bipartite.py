from sacorg.algorithms.miller_harrison import miller_harrison
from sacorg.algorithms.chen_diaconis_holmes_liu import chen_diaconis_holmes_liu

def count(deg_seq1, deg_seq2, method, num_of_samples=1000, verbose=False):

    if method == "MH":
        result = miller_harrison.count(p=deg_seq1, q=deg_seq2, verbose=verbose)
        return result

    if method == "CDHL":
        estimate = chen_diaconis_holmes_liu.count(p=deg_seq1, q=deg_seq2, num_of_samples=num_of_samples, verbose=verbose)
        return estimate


def get_sample(deg_seq1, deg_seq2, method, num_of_samples, iteration=-1, verbose=False):

    if method == "MH":
        return miller_harrison.get_sample(p=deg_seq1, q=deg_seq2, num_of_samples=num_of_samples, verbose=verbose)

    if method == "CDHL":
        return chen_diaconis_holmes_liu.get_sample(p=deg_seq1, q=deg_seq2, num_of_samples=num_of_samples, verbose=verbose)