#!/usr/bin/env python

import copy
class testc:
    x=0

def main():
    test=testc()
    test.x=4
    test2=copy.deepcopy(test)
    test2.x=5
    print(test.x)
    print(test2.x)



if __name__ == '__main__':
    main()


print('class vs instance variables?')
test=testc()
test2=testc()
test2.x=1
print(test.x)
print(test2.x)




import scipy

a=scipy.array([[1,2],[3,4]])
print(a)
for i in a:
    i=i+1
    print(i)
print(a)


from datetime import datetime
from datetime import timedelta
now=datetime.now()
print(now)
hour=timedelta(hours=1)
print(hour)
print(now+hour)

print('wait 5 seconds')
print(datetime.now())
stopTime=now+timedelta(seconds=5)
while True:
    if datetime.now()>=stopTime:
        print(datetime.now())
        break



while False:
    print("so lonenly")
else:
    print("looks weird")

test=True
while test:
    print("so lonely 2")
    test=False
else:
    print("looks weird 2")

a=[1,2,3]
a.append([4,5])
print(a)

##print(int([1.3,4,4.4]))

import math
import random
test=[]
print('testing...')
for i in range(10000):
    test.append(int(math.sqrt(random.uniform(1,10**2)))-1)
print("0 "+str(test.count(0)))
print("1 "+str(test.count(1)))
print("2 "+str(test.count(2)))
print("3 "+str(test.count(3)))
print("4 "+str(test.count(4)))
print("5 "+str(test.count(5)))
print("6 "+str(test.count(6)))
print("7 "+str(test.count(7)))
print("8 "+str(test.count(8)))
print("9 "+str(test.count(9)))
print("10 "+str(test.count(10)))

test=[]
print('testing2...')
for i in range(10000):
    test.append(random.randint(0,10))
print("0 "+str(test.count(0)))
print("1 "+str(test.count(1)))
print("2 "+str(test.count(2)))
print("3 "+str(test.count(3)))
print("4 "+str(test.count(4)))
print("5 "+str(test.count(5)))
print("6 "+str(test.count(6)))
print("7 "+str(test.count(7)))
print("8 "+str(test.count(8)))
print("9 "+str(test.count(9)))
print("10 "+str(test.count(10)))



for i in range(0):
    print(i)
    print("iiiiiiiiiiiiiiii")

print("tester")
i=1
i+=1
print(i)


print('weird class test')
class gjsweird:
    ID=0
    id=0

    def __init__(self):
        gjsweird.ID+=1
        self.id=gjsweird.ID

classID1=gjsweird()
classID2=gjsweird()
classID3=gjsweird()

print(classID1.id)
print(classID2.id)
print(classID3.id)

print('random test')
import random
for i in range(10):
    print(random.normalvariate(0,1))

print('instance variable pointers?')
class testerdetest:
    id=1
test1=testerdetest()
test2=test1
test3=copy.copy(test1)
test2.id=2
print(test1.id)
print(test2.id)
print(test3.id)
