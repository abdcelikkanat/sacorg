import unittest
import sys
sys.path.append('../sacorg')
from simple import MyAlg, MCMC
import numpy as np
import matplotlib.pyplot as plt


class Test(unittest.TestCase):

    def test_mcmc_uniformity(self):

        # ------------------------------------------
        d1 = np.asarray([2, 2, 2, 2])
        d2 = np.asarray([2, 2, 2, 2, 2, 2, 2])
        d3 = np.asarray([3, 3, 3, 3, 3, 3])
        d4 = np.asarray([4, 4, 3, 3, 2, 2, 2])
        d5 = np.asarray([3, 2, 2, 1, 1, 1])
        # ------------------------------------------

        myalg = MyAlg()
        mcmc = MCMC()

        # ------------------------------------------
        # Determine a sequence
        d = d5
        total = myalg.count(deg_seq=d)
        number_of_samples = total*1000
        # ------------------------------------------

        samples = mcmc.get_sample(deg_seq=d, num_of_samples=number_of_samples, verbose=True)

        histogram = {}
        for i in range(number_of_samples):
            key = repr(np.asarray(samples[i]))
            if key in histogram:
                histogram[key] += 1
            else:
                histogram[key] = 1

        values = []
        for key, value in enumerate(histogram.values()):
            for j in range(value):
                values.append(key)

        # Plot the histogram
        plt.hist(x=values, bins=total)
        plt.show()
