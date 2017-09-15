import sys
import os.path
sys.path.append('../sacorg')
from simple import MyAlg, BlitzsteinDiaconis
import numpy as np


d = np.asarray([2,2,2,1,1,1,1])
d2 = np.asarray([2,2,2,2,2])
d = np.asarray([1,1,2,2,2,1,1])

myalg = MyAlg()
#print myalg.count(deg_seq=d)

#a = np.sort([4,1,2,1,3])[::-1]
#print a

#bd = BlitzsteinDiaconis()
#d = np.asarray([1,1,2,2,2,1,1])
#print bd.get_sample(deg_seq=d, num_of_samples=1)

d = np.asarray([2,7,1,8,2,8,1,8,2,8,4,5,9,0,4,5])
d = np.asarray([7, 8, 5, 1, 1, 2, 8, 10, 4, 2, 4, 5, 3, 6, 7, 3, 2, 7, 6, 1, 2, 9, 6, 1, 3, 4, 6, 3, 3, 3, 2, 4, 4])
d = np.asarray([2,5,7, 4, 14, 4, 4, 14, 3, 5, 7, 9, 6, 6, 4, 8, 6, 7, 4, 4, 7, 3, 4, 7])
print myalg.count(deg_seq=d)