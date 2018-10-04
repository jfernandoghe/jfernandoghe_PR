import pylab
import numpy as np
import random
import math
from numpy.lib.scimath import logn


# Data
bou = 50
st  = 1
x   = np.linspace(-bou, bou, bou*2)    #100 linearly spaced numbers
y   = (pow(x, 3))-(np.log(x))           #Computing the values of x^3-log(x)
sp  = random.randrange(-bou, bou, 1)

#Program
while True:
    m=(x[bou+sp]-x[bou+sp+st])/(y[bou+sp]-y[bou+sp+st])
    print (m)
    break


#print (sp, x[sp], y[sp])
pylab.plot(x,y)                         #x^3-log(x)
pylab.plot(x[bou+sp], y[bou+sp], 'c*')
pylab.show()                            #Show the plot

