# A Machine Learning Algorithm Optimization Based On Amazon Apache Spark
# Model 2 - Average Weights from Different Partitions At the Last

import sys
from random import random
from operator import add
from pyspark import SparkContext

if __name__ == "__main__":
    """
        Usage: <file name> [partitions]
    """
    # create SparkContext
    sc = SparkContext(appName="perceptron")

    # read how many partitions from the command line
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print("partitions = ", partitions)

    # the parameter optimized based on COMP 520 project
    # the number of iterations
    maxIterations = 1000
    # the learning rate
    learningRate = 0.001
    # the number of attributes
    numWeights = 7
    # inital weights all 0.001
    # the ith cluster has its own inital weights w[i]
    w = [[0.001] * numWeights] * partitions

    # function: update weights
    # input parameters:
    #   index: the cluster index number
    #   it: iterations for records of one partition
    # output: the updated weights learnt by one partition data
    def update(index, it):
        w0 = w[index]
        for x in it:
            t = 0.0
            for ii in range(numWeights):
                t += x[ii] * w0[ii]
            if t >= 0.0:
                ycap = 1.0
            else:
                ycap = -1.0
            cof = learningRate * (x[numWeights] - ycap)
            for ii in range(numWeights):
                w0[ii] += cof * x[ii]
        yield w0

    # funciton: calculate errors
    # input parameters:
    #   x: one record of the dataset
    #   w: weights
    #   n: the number of data in the dataset
    # ouptut: the error between the real y and the predicted y from weights
    def updateIt(index, it):
        w0 = w[index]
        for kk in range(maxIterations):
            for x in it:
                t = 0.0
                for ii in range(numWeights):
                    t += x[ii] * w0[ii]
                if t >= 0.0:
                    ycap = 1.0
                else:
                    ycap = -1.0
                cof = learningRate * (x[numWeights] - ycap)
                for ii in range(numWeights):
                    w0[ii] += cof * x[ii]
        yield w0

    # function: string to floats
    # input parameters:
    #   x: one record of the dataset
    #   n: the number of data in the dataset
    # output: convert a string to float
    def error(x, w, n):
        t = 0.0
        for jj in range(n):
            t = t + x[jj] * w[jj]
        if t >= 0:
            ycap = 1
        else:
            ycap = -1
        return abs(x[n] - ycap)/2.0
    
    # function: string to floats
    # input parameters:
    #   x: one record of the dataset
    #   n: the number of data in the dataset
    # output: convert a string to float
    def s2f(x, n):
        x2 = [0] * n
        for jj in range(n):
            x2[jj] = float(x[jj])
        return x2

    # read data from disk and partition to the clusters on Amazon EC2 system
    train = sc.textFile("file:///home/ec2-user/pj512bigdata/train.csv", partitions).map(lambda x: s2f(x.split(","), numWeights + 1)).cache()
    valid = sc.textFile("file:///home/ec2-user/pj512bigdata/valid.csv", partitions).map(lambda x: s2f(x.split(","), numWeights + 1)).cache()
    test  = sc.textFile("file:///home/ec2-user/pj512bigdata/test.csv",  partitions).map(lambda x: s2f(x.split(","), numWeights + 1)).cache()

    # get # of records for each dataset
    train_num = train.count()
    valid_num = valid.count()
    test_num  = test.count()

    # leraing weights
    w = train.mapPartitionsWithIndex(updateIt).collect()
    
    # average weights
    avgW = [0] * numWeights
    for jj in range(numWeights):
        for kk in range(partitions):
            avgW[jj] += w[kk][jj] / (partitions * 1.0)

    # calculate errors
    errTrain = 100.0 / train_num * train.map(lambda x: error(x, avgW, numWeights)).reduce(lambda x1, x2: x1 + x2)
    errValid = 100.0 / valid_num * valid.map(lambda x: error(x, avgW, numWeights)).reduce(lambda x1, x2: x1 + x2)
    errTest  = 100.0 /  test_num *  test.map(lambda x: error(x, avgW, numWeights)).reduce(lambda x1, x2: x1 + x2)
    print(errTrain,errValid,errTest)

    # close SparkContext
    sc.stop()
