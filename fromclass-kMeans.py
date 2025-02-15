'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
# Edited slighlty by Clif Baldwin, January 2023

# Document the code so a non-Python programmer can understand what it is doing
import numpy as np
import pandas as pd
import math

# You can rewrite loadDataSet() 
#  or replace it with what I did in the K-NN Python example
def loadDataSet(fileName):
    dataMat = []                
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

# This function should work for you.
# Document what it is doing specifically with inline comments
# If you want to rewrite it, go ahead
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))[0,0]    

# This function is provided by Peter Harrington via his GitHub page
# You have to modify it to make it work
# And then document what each line is doing with inline comments
def createCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n): 
        minJ = min(dataSet[:,j]) 
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

# This function is provided by Peter Harrington via his GitHub page
# You have to modify it to make it work
# And then document what each line is doing with inline comments
def kMeans(dataSet, k, distEclud, createCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distEclud(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
            
    return centroids, clusterAssment


# I modified the following code from a different source.
# The function showPlt() should take your dataset, compute the K clusters, and plot the results
# It should work, and you can use it as written.
# You do not have to provide inline comments for this function.
import matplotlib
import matplotlib.pyplot as plt
def showPlt(datMat, alg=kMeans, numClust=3):
    myCentroids, clustAssing = alg(datMat, numClust)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

print("Thinh")
