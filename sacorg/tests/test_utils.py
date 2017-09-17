import unittest
from sacorg.utils import *


class Test(unittest.TestCase):

    def test_vector_of_counts(self):
        """ Test the vector_of_counts method """

        d1 = np.asarray([2, 3, 1, 2, 3, 3, 4, 2, 1, 3, 4])
        output = vector_of_counts(deg_seq=d1)
        actual = np.asarray([2, 3, 4, 2])
        self.assertListEqual(actual.tolist(), output.tolist())

        d2 = np.asarray([1, 2, 2, 1, 2])
        output = vector_of_counts(deg_seq=d2, dim=3)
        actual = np.asarray([2, 3, 0])
        self.assertListEqual(actual.tolist(), output.tolist())