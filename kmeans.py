from numpy import load, savez, array, arange, tile, random, delete, exp, dot
# import math
import numpy as np


def kmeans(X, k):
        
    '''
    KMEANS ALGORITHM procedure:
    1 - Initialize the center of each cluster to a different randomly selected training pattern.
    2 - Assign each training pattern to the nearest cluster. This can be accomplished by calculating
        the Euclidean distances between the training patterns and the cluster centers.
    3 - When all training patterns are assigned, calculate the average position for each cluster center.
        They then become new cluster centers.
    4 - Repeat steps 2 and 3, until the cluster centers do not change during the subsquent iterations.
    

    Performs k-means clustering for N-D input
    Arguments:
        X {ndarray} -- A MxN array of inputs
        k {int} -- Number of clusters
    
    Returns:
        ndarray -- A kxN array of final cluster centers
    '''

    # randomly select initial cluster center from input data
    centers = X[np.random.randint(X.shape[0], size=k)]
    print(X.shape[0])
    print("cluster centros iniciais", centers )
    prevCenters = centers.copy()
    stds = np.zeros(k)
    converged = False

    while not converged:
        
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        
        distances = np.abs(X[:, np.newaxis] - centers[np.newaxis, :])
        print("\n\n\n\n\n\ndistancias", distances)

        dist = np.sum(distances, axis=2)
        print("soma das distancias", dist)
        
        closestCenters = np.argmin(dist, axis=1)

        # find the cluster center that's closest to each point
        ClusterForPoint = []
        for point in range(len(X)):
            ClusterForPoint.append([closestCenters[point], X[point]])
            # print(closestCenters[point], X[point])
        
        # update centers by taking the mean of all of the points assigned to that cluster
        newCenters = [] 
        for n in range(k):
            cluster = []
            for c in range(len(ClusterForPoint)):
                if ClusterForPoint[c][0] == n:
                    cluster.append(ClusterForPoint[c][1])
            
            cluster = array(cluster)

            meanPoints = np.mean(cluster, axis=0)
            newCenters.append(meanPoints)

        # converge if centers haven't moved
        centers = array(newCenters)
        # print("novos centros", centers)
        
        converged = np.linalg.norm(centers - prevCenters) < 1e-6
        prevCenters = centers.copy()
        
        # Gráfico 2D
        # plt.plot(X[:, 0], X[:, 1], 'bx')
        # plt.plot(centers[:, 0], centers[:, 1], 'ro')
        # plt.show()
        
        # Gráfico 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r')
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='^')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

    ''' 
        Standard deviation 
    '''

    print("final centers", centers)
    print("\n\n -- Standard deviation\n\n")

    distances = np.abs(X[:, np.newaxis] - centers[np.newaxis, :])
    # print(distances)
    
    dist = np.sum(distances, axis=2)
    # print(dist)

    closestCenters = np.argmin(dist, axis=1)
    # print(closestCenters)

    # find the cluster center that's closest to each point
    ClusterForPoint = []
    for point in range(len(X)):
        ClusterForPoint.append([closestCenters[point], X[point]])
        # print(closestCenters[point], X[point])
    
    clustersWithNoPoints = []
    for n in range(k):
        cluster = []
        for c in range(len(ClusterForPoint)):
            if ClusterForPoint[c][0] == n:
                cluster.append(ClusterForPoint[c][1])
        cluster = array(cluster)

        if len(cluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(n)
            continue
        else:
            stds[n] = np.std(cluster)

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for n in range(k):
            if n not in clustersWithNoPoints:
                for c in range(len(ClusterForPoint)):
                    if ClusterForPoint[c][0] == n:
                        pointsToAverage.append(ClusterForPoint[c][1])

        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    print("standard deviation vector", stds)

    return centers, stds
