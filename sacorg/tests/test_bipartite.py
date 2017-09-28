import unittest

from sacorg.algorithms import bipartite
from sacorg.utils import *


class Test(unittest.TestCase):

    def test_miller_harrison_arbitrary1(self):
        """ Test the values for bipartite graphs """

        actual_result = 8

        seq1 = np.asarray([2, 2, 1, 1])
        seq2 = np.asarray([3, 2, 1])
        computed_result = bipartite.count(deg_seq1=seq1, deg_seq2=seq2, method="MH", verbose=False)

        self.assertEqual(actual_result, computed_result)

    def test_miller_harrison_arbitrary2(self):
        """ Test the values for bipartite graphs """

        actual_result = 839926782939601640

        seq1 = np.asarray([14, 14, 14, 12, 5, 13, 9, 11, 11, 11, 11, 11, 7, 8, 8, 7, 2, 4, 2, 3, 2, 2, 2])
        seq2 = np.asarray([21, 19, 18, 19, 14, 15, 12, 15, 12, 12, 12, 5, 4, 4, 1])
        computed_result = bipartite.count(deg_seq1=seq1, deg_seq2=seq2, method="MH", verbose=False)

        self.assertEqual(actual_result, computed_result)


if __name__ == "__main__":
    unittest.main()