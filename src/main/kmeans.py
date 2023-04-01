import matplotlib.pyplot as plot
import numpy as np
import math
import random


def euclidean_distance(a, b):
    """
    Function to calculate the Euclidean distance between two data points in a 2D space.
    #Euclidean distance (l2 norm)
    :param a: each value from dataset
    :param b: each value from centroids
    :return:
    """
    return math.sqrt(math.pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2))


# Step 1
def closestCentroid(x, centroids):
    """
    Function to assign each data point to the cluster with the closest centroid.
    :param x: 2D array. i.e., dataset
    :param centroids: initial centroids to being with and updated for every iteration.
    :return: returning an array of data points to the cluster with closest centroid.
    """
    assignments = []
    for i in x:
        # distance between one data point and centroids
        distance = []
        for j in centroids:
            distance.append(euclidean_distance(i, j))
            # assign each data point to the cluster with closest centroid
        assignments.append(np.argmin(distance))
    return np.array(assignments)


# Step 2
def updateCentroid(x, clusters, K):
    """
    Function to update the centroids of each cluster based on the mean of all the data points in that cluster.
    :param x: 2D array. i.e., dataset
    :param clusters:
    :param K: 2
    :return: updated centroid
    """
    new_centroids = []
    for c in range(K):
        # Update the cluster centroid with the average of all points in this cluster
        cluster_mean_x = x[:, 0][clusters == c].mean()
        cluster_mean_y = x[:, 1][clusters == c].mean()
        new_centroids.append([cluster_mean_x, cluster_mean_y])
    return np.array(new_centroids)


# 2-d kmeans
def kmeans(x, K):
    """
    Function to perform 2D k-means clustering on a given dataset x with K clusters.
    :param x: 2D array. i.e., dataset
    :param K: Cluster
    """
    centroids = initCentroid(K)
    print('Initialized centroids: {}'.format(centroids))
    for i in range(10):
        clusters = closestCentroid(x, centroids)
        displayPlot(x, centroids)
        centroids = updateCentroid(x, clusters, K)
        print('Iteration: {}, Centroids: {}'.format(i, centroids))


def initCentroid(K):
    """
    Function to initialize the centroids of K clusters with random x,y coordinates
    :param K: 2
    :return: calculated value from x & y coordinates.
    """
    x_coordinates = []
    y_coordinates = []
    # Assign random integers to x & y coordinates for 2 iterations(K times).
    for _ in range(K):
        x_coordinates.append(random.randint(0, 10))
        y_coordinates.append(random.randint(0, 12))
    centroid = np.array([x_coordinates, y_coordinates])
    return centroid


def displayPlot(X, Y):
    """
    Function to display a scatter plot of the data points and centroids.
    :param X: data set
    :param Y: centroids
    """
    # define the range of plot
    plot.axis([0, 10, 0, 10])
    plot.xlabel('x')
    plot.ylabel('y')
    plot.plot(X[:, 0], X[:, 1], 'ro')
    plot.plot(Y[:, 0], Y[:, 1], 'bo')
    plot.show()


# Number of clusters/centroids
K = 2
# Given input 2-D array
x = np.array([[2, 4],
              [1.7, 2.8],
              [7, 8],
              [8.6, 8],
              [3.4, 1.5],
              [9, 11]])
# Calling kmeans classifier function
kmeans(x, K)
