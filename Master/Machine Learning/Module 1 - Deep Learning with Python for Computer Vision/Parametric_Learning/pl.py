# -*- coding: utf-8 -*-
############################################################
#       Coded by:
#       José Fernando González Herrera
#       jfernandoghe@gmail.com
#       March, 2018 - Mexico
############################################################
# python pl.py

import numpy as np
import matplotlib.pyplot as plt

n = 120
k = 3   # 3 Classes [labels]
d = 2
labels = ['Cat', 'Dog', 'Panda']

cx = np.random.randint(30, size=(d, n/2))

dx1 = 30+np.random.randint(40, size=(d/2, n/2))
dx2 = 30+np.random.randint(40, size=(d/2, n/2))

dx = (np.concatenate((dx1,dx2), axis=0))

x = (np.concatenate((cx,dx), axis=1))

plt.plot(cx[0,:], cx[1,:], 'bs', dx[0,:], dx[1,:], 'go',)
plt.show()

W = np.random.randn(k, d)
b = np.random.randn(k,1)

scores = (W[1,1]*(x[1,1])) + b[1,1]

#scores = W[1,1].dot(x[1,1]) + b[1,1]

#print(cx[1,:])
#print('dx', dx)
print('cx size', np.shape(cx))
print('dx1 size', np.shape(dx1))
print('dx2 size', np.shape(dx2))
print('dx size', np.shape(dx))
print('xi size', np.shape(x))
print('xi size', len(x))
print('xi size', x.size/len(x))
#print('scores', scores)
print('scores len', len(scores))
print('scores size', x.size/len(scores))