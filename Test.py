import numpy as np
a = np.arange(3*4*5*6).reshape((3,4,5,6))
b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
print np.dot(a, b)[2,3,3,1,2,2]
print sum(a[2,3,3,:] * b[1,2,:,2])