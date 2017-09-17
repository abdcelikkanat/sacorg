from sacorg.utils import *


def is_graphical(deg_seq, method="Erdos-Gallai", is_sorted=False):
    """
        Checks whether given degree sequence d is graphical or not
        :param deg_seq: given degree sequence
        :param method: method to check given degree sequence is graphical or not
        :param is_sorted
        :return: boolean value representing whether given sequence is graphical or not
    """
    # Copy the given degree sequence
    d = np.asarray(deg_seq).copy()

    # If the length of the sequence is 0, it is graphical
    if len(d) == 0:
        return True

    # All degrees must be greater than zero
    if np.any(d < 0):
        return False

    # The sum of degrees must be even
    if sum(d) % 2:
        return False

    # Sort the sequence in non-increasing order
    if is_sorted is False:
        d = np.sort(d)[::-1]

    # Get the length of the sequence
    N = len(d)

    """
        Implementation of Erdos-Gallai Theorem
    """
    if method == "Erdos-Gallai":

        # Check all n conditions
        for k in range(1, N + 1):
            if sum(d[0:k]) > k * (k - 1) + sum([min(d_i, k) for d_i in d[k:N]]):
                return False
        # Return true, if all conditions are satisfied
        return True