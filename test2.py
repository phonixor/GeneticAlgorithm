#!/usr/bin/env python


import numpy as np
from scipy.optimize import fsolve

tmpFunc = lambda xIn: np.array( [(xIn[0]-4)**2 + xIn[1], (xIn[1]-5)**2 - xIn[2], (xIn[2]-7)**3 + xIn[0] ] )

x0 = [3,4,5]
xFinal = fsolve(tmpFunc, x0 )

print(xFinal)


print(tmpFunc([1,1,1]))



import scipy
test=[1,2,3,4,0,0,0,0]
print(test)

testlen=len(test)/2
print(testlen)

print(test[0:testlen:1])
print(test[testlen-1::-1])

test[testlen:]=test[testlen-1::-1]
print(test)




print('--------------')

correctlyPredicted=[]
variables=scipy.array([[1,-2,3,4],[1,-1,2,-2]])
i=0
correctlyPredicted.append(sum(x>0 for x in variables[i,:]))
print(variables[i,:])
print(correctlyPredicted)
print(sum(variables[i,:]>0))

correctlyPredicted=scipy.array([-1, -1, 129, -1, 100, -1, -1, 44, 132, 132, 132, 132, -1, -1, -1, -1, 103, 56, -1, 98, 81, 70, 39, -1, 102, 78, 79, 98]).astype(float)
print('correctly predicted #: '+str(correctlyPredicted))
test=correctlyPredicted/132
print(test)
print('correctly predicted %: '+str(((correctlyPredicted/132)*100).astype(int)))



print('----------------')
print(not(1))
print(not(0))