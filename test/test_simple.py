import unittest
import sys
import os.path
sys.path.append('../sacorg')
from simple import MyAlg
import numpy as np


class Test(unittest.TestCase):

    def test_vector_of_counts(self):
        """ Test the vector_of_counts method """
        myalg = MyAlg()

        d1 = np.asarray([2, 3, 1, 2, 3, 3, 4, 2, 1, 3, 4])
        output = myalg.vector_of_counts(deg_seq=d1)
        actual = np.asarray([2, 3, 4, 2])
        self.assertListEqual(actual.tolist(), output.tolist())

        d2 = np.asarray([1, 2, 2, 1, 2])
        output = myalg.vector_of_counts(deg_seq=d2, dim=3)
        actual = np.asarray([2, 3, 0])
        self.assertListEqual(actual.tolist(), output.tolist())

    def test_myalg_2regular(self):
        """ Test the values for 2-regular graphs """
        # www.oeis.org
        # A001205 Number of undirected 2-regular labeled graphs
        actual_results = [1, 0, 0, 1, 3, 12, 70, 465, 3507, 30016, 286884, 3026655, 34944085, 438263364, 5933502822,
                          86248951243, 1339751921865, 22148051088480, 388246725873208, 7193423109763089,
                          140462355821628771, 2883013994348484940]
        myalg = MyAlg()
        for n in range(0, len(actual_results)):
            seq = np.ones(n, dtype=np.int)*2

            computed_result = myalg.count(deg_seq=seq, verbose=False)
            self.assertEqual(actual_results[n], computed_result)

    def test_myalg_3regular(self):
        """ Test the values for 3-regular graphs """
        # www.oeis.org
        # A002829 Number of trivalent (or cubic) labeled graphs with 2n nodes
        actual_results = [1, 0, 1, 70, 19355, 11180820, 11555272575, 19506631814670, 50262958713792825,
                          187747837889699887800, 976273961160363172131825, 6840300875426184026353242750,
                          62870315446244013091262178375075, 741227949070136911068308523257857500]
        myalg = MyAlg()
        for n in range(0, len(actual_results)):
            seq = np.ones(2*n, dtype=np.int)*3

            computed_result = myalg.count(deg_seq=seq, verbose=False)
            self.assertEqual(actual_results[n], computed_result)

    def test_myalg_4regular(self):
        """ Test the values for 4-regular graphs """
        # www.oeis.org
        # A005815 Number of 4-valent labeled graphs with n nodes.
        actual_results = [1, 0, 0, 0, 0, 1, 15, 465, 19355, 1024380, 66462606, 5188453830, 480413921130,
                          52113376310985, 6551246596501035, 945313907253606891, 155243722248524067795,
                          28797220460586826422720]
        myalg = MyAlg()
        for n in range(0, len(actual_results)):
            seq = np.ones(n, dtype=np.int)*4

            computed_result = myalg.count(deg_seq=seq, verbose=False)
            self.assertEqual(actual_results[n], computed_result)

    def test_myalg_1(self):
        """ Test an arbitrary sequence """
        myalg = MyAlg()
        seq = np.asarray([2, 2, 2])
        result = myalg.count(deg_seq=seq, verbose=False)
        self.assertEqual(result, 1)

    def test_myalg_2(self):
        """ Test an arbitrary sequence """
        myalg = MyAlg()
        seq = np.asarray([2,2,3,1,7,2,5,3,5])
        result = myalg.count(deg_seq=seq, verbose=False)
        self.assertEqual(result, 215)

    def test_myalg_3(self):
        """ Test an arbitrary sequence """
        myalg = MyAlg()
        seq = np.asarray([4,6,6,5,2,1,3,8,4,1,4])
        result = myalg.count(deg_seq=seq, verbose=False)
        self.assertEqual(result, 117697)

    def test_myalg_4(self):
        """ Test an arbitrary sequence """
        myalg = MyAlg()
        seq = np.asarray([4,2,5,2,2,3])
        result = myalg.count(deg_seq=seq, verbose=False)
        self.assertEqual(result, 3)

    def test_myalg_5(self):
        """ Test an arbitrary sequence """
        myalg = MyAlg()
        seq = np.asarray([2,4,2,4,5,4,1])
        result = myalg.count(deg_seq=seq, verbose=False)
        self.assertEqual(result, 12)

if __name__ == "__main__":
    unittest.main()