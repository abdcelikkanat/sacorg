from sacorg.utils import *


def is_bigraphic(p, q, method="Gale-Ryser"):
    """
        Checks whether given degree sequences of partite sets (p,q) is bigraphic or not
        :param p: degree sequence of a parite set
        :param q: degree sequence of the other parite set
        :param method: method to check given degree sequence is graphical or not
        :param is_sorted
        :return: boolean value representing whether given sequence is graphical or not
    """
    # Copy the given degree sequences
    d1 = np.asarray(p).copy()
    d2 = np.asarray(q).copy()

    # If the length of the sequences is 0, it is bigraphic
    if len(d1) == 0 and len(d2) == 0:
        return True

    # All degrees must be greater than zero
    if np.any(d1 < 0) or np.any(d2 < 0):
        return False

    # The sum of degree sequences must be equal
    if sum(d1) != sum(d2):
        return False

    # Maximum of a sequence cannot be larger than the number of non-zero elements in the other sequence
    if np.sum(d1 > 0) < max(d2) or np.sum(d2 > 0) < max(d1):
        return False

    """
        Implementation of Gale-Ryser Theorem
    """
    if method == "Gale-Ryser":

        # Consider only the elements greater than zero in the sequences
        d1 = np.asarray([x for x in d1 if x > 0])
        d2 = np.asarray([x for x in d2 if x > 0])

        # Get the length of the sequence d1
        N1 = len(d1)

        # Find the conjugate of the sequence d2 with the size of N1
        conj_d2 = [np.sum(d2 >= x) for x in range(1, N1+1)]

        # Checkthe conditions
        for k in range(1, N1+1):
            if sum(d1[0:k]) > sum(conj_d2[0:k]):
                return False
        # Return true, if all conditions are satisfied
        return True