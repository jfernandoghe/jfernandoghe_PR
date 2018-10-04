from pyflann import *
import numpy as np
from sklearn.metrics import classification_report

dataset = np.array(
    [[1.,   1,      1,      2,  3],
     [10,   10,     10,     3,  2],
     [100,  100,    2,      30, 1]
     ])
testset = np.array(
    [[1.,   1,  1,  1,  1],
     [90,   90, 10, 10, 1]
     ])
#dataset = np.random.rand(10000, 128)
#testset = np.random.rand(1000, 128)
flann = FLANN()
result, dists = flann.nn(
    dataset, testset[1,:], 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
print result
print dists
#print("dists", dists)
#print(classification_report(dataset, testset))
