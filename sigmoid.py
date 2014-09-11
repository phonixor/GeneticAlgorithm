# just display the sigmoid function for in report

import math
def sigma(value):
    """ function that returns a value between 0-1 with a slope around when value=0
        see the model for more information

    """
    return 0.5*((value/math.sqrt((value**2)+1))+1)


import pylab
import scipy

time=scipy.linspace(-10,10,10000)
sig=[]
for i in range(len(time)):
    sig.append(sigma(time[i]))


pylab.figure()
pylab.plot(time,sig)
pylab.title('sigma')
pylab.show()