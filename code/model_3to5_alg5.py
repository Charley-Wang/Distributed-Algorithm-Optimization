# A Machine Learning Algorithm Optimization Based On Amazon Apache Spark
# Model 3 to 5 - Mix the idea of Model 1 and 2

import sys
from random import random
from operator import add
from pyspark import SparkContext

if __name__ == "__main__":
    """
        Usage: <file name> [partitions] [subIterations]
    """
   
    # create SparkContext 
    sc = SparkContext(appName="perceptron")

    # read parameters from the command line
    partitions = int(sys.argv[1])
    subIterations = int(sys.argv[2])
    print("partitions = ", partitions)
    print("subIterations = ", subIterations)

    # the parameter optimized based on COMP 520 project
    # the number of iterations for outer cycle: maxIterations
    # the number of iterations for inner cycle: subIterations
    #     which is # of iterations in each partition
    maxIterations = 1000 / subIterations
    # the learning rate 
    learningRate = 0.001
    # the number of attributes
    numWeights = 7
    # inital weights all 0.001
    # the ith cluster has its own inital weights w[i]
    w = [[0.001] * numWeights] * partitions

    # function: update weights with subIterations
    # input parameters:
    #   index: the cluster index number
    #   it: iterations for records of one partition
    # output: the updated weights learnt by one partition data
    def updateIt(index, it):
        w0 = w[index]
        for kk in range(subIterations):
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

    # inital errors
    errt = [0.0] * maxIterations
    errv = [0.0] * maxIterations
    errx = [0.0] * maxIterations

    # leraing weights
    for ii in range(maxIterations):
        # inner cycle for updating weights
        w = train.mapPartitionsWithIndex(updateIt).collect()
        # average weights from different partitions
        avgW = [0] * numWeights
        for jj in range(numWeights):
            for kk in range(partitions):
                avgW[jj] += w[kk][jj] / (partitions * 1.0)
        # calculate errors for train, valid, and test datasets
        errTrain = 100.0 / train_num * train.map(lambda x: error(x, avgW, numWeights)).reduce(lambda x1, x2: x1 + x2)
        errValid = 100.0 / valid_num * valid.map(lambda x: error(x, avgW, numWeights)).reduce(lambda x1, x2: x1 + x2)
        errTest  = 100.0 /  test_num *  test.map(lambda x: error(x, avgW, numWeights)).reduce(lambda x1, x2: x1 + x2)
        errt[ii] = errTrain
        errv[ii] = errValid
        errx[ii] = errTest     
        print(ii,errTrain,errValid,errTest)
        # set new weights w for the next interation
        w = [avgW] * partitions

    # calculate the last 50 iterations' average errors
    avgt = 0.0
    avgv = 0.0
    avgx = 0.0
    avgN = 10
    for ii in range(maxIterations - avgN, maxIterations):
        avgt += errt[ii] / (1.0 * avgN)
        avgv += errv[ii] / (1.0 * avgN)
        avgx += errx[ii] / (1.0 * avgN)
    print(avgt,avgv,avgx)

    # close SparkContext
    sc.stop()
