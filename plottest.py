


import pylab
import scipy




##print('simple')
##pylab.plot([1,2,3],[1,2,3])
##pylab.show()
##
##
##print('double')
### pylab.plot([1,2,3],[[2,3,4],[1,2,3]]) # wrong dimensions!
##pylab.plot([1,2,3],[[1,2],[2,3],[3,4]])
##pylab.show()
##
##print('double dots')
##pylab.plot([1,2,3],[[1,2],[2,3],[3,4]], 'b.')
##pylab.show()


##
##print('array')
##test=[[1,2],[2,3],[3,4]]
##testArray=scipy.array(test)
##pylab.plot([1,2,3],testArray)
##pylab.show()
##
##print(testArray.shape)
##
##
##test1=[1,2,3]
##test2=[2,3,4]
##
##print('array created differently')
##test=[]
##test.append(test1)
##test.append(test2)
##testArray=scipy.array(test)
##print(testArray.shape)
##testArray=scipy.transpose(testArray)
##print(testArray.shape)
##pylab.plot([1,2,3],testArray)
##pylab.show()


##x=[1,2,3,4,5]
##yr=[2,4,6,7,8]
##yc=[1,2,2,2,5]
##yg=[1,3,4,6,1]
##
##pylab.bar(x, yr, color='#88aa33', align='center', label='histogram')
##pylab.plot(x, yc, 'bo-', label='cumulative')
##pylab.plot(x, yg, 'ro-',label='2nd set of cookies')
##pylab.legend()
##pylab.show()



##import matplotlib.pyplot as plt
##fig = plt.figure()
##image1=fig.add_subplot(111)
##t = scipy.arange(0,10,1)
##data1=t*2
##data2*t*3
##
##






##
##
##from pylab import plot, show, ylim, yticks
##import scipy
##
##
##
####.numerix import sin, cos, exp, pi, arange
##
##
##t = scipy.arange(0.0, 2.0, 0.01)
##s1 = t
##s2 = t*2
##s3 = t*3
##s4 = t*4
##
####plot(t, s1, t, s2+1, t, s3+2, t, s4+3, color='k')
##pylab.plot(t, s1, t, s2+1, t, s3+2, t, s4+3)
##ylim(-1,4)
##yticks(scipy.arange(4), ['S1', 'S2', 'S3', 'S4'])
##
##show()



##x=[1,2,3,4,5]
##yr=[2,4,6,7,8]
##yc=[1,2,2,2,5]
##yg=[1,3,4,6,1]
##
##pylab.figure()
##pylab.title('BIG MOTHAH ...')
##pylab.bar(x, yr, color='#88aa33', align='center', label='histogram') # completely ignored!!! subplots are wonderful!
##pylab.subplot(1,2,1)
##pylab.title('test1')
##pylab.plot(x, yc, 'bo-')
##pylab.subplot(1,2,2)
##pylab.title('test2')
##pylab.plot(x, yg, 'ro-')
##pylab.legend()
##pylab.show()




import matplotlib.pyplot as plt
import numpy as np

x,y = np.random.randn(2,100)
fig = plt.figure()

fig.subplots_adjust(left=0.25, bottom=0.25)
##axtitle=fig.add_axes([0.8, 0.025, 0.1, 0.04])
axtitle=fig.text(0.5,0.5,'lalalalalala')


ax1 = fig.add_subplot(211)
plt.ylim(1)
ax1.xcorr(x, y, usevlines=True, maxlags=50, normed=True, lw=2)
ax1.grid(True)
ax1.axhline(0, color='black', lw=2)


ax2 = fig.add_subplot(212, sharex=ax1)
ax2.acorr(x, usevlines=True, normed=True, maxlags=50, lw=2)
ax2.grid(True)
ax2.axhline(0, color='black', lw=2)

##fig.show() # looks nicer from a coding perspective... but crashes ... ... so fuck it!
plt.show()



