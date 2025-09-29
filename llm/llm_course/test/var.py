import random
import math
import numpy

allNum = 12
recurrentTime = 100000

def myVar(num):
    avg = numpy.average(num)
    avgArr = numpy.full(len(num), avg)
    tmp = numpy.square(numpy.subtract(num, avgArr))
    return numpy.sum(tmp) / (len(num) - 1)

for j in range(2, allNum):
    res = 0
    for i in range(0, recurrentTime):
        numbers = random.choices(range(11), k=j)
        Bar = numpy.average(numbers)
        Var = myVar(numbers)
        res = Var + res
    print(res / recurrentTime)