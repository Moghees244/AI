import random
import pandas as pd
import matplotlib.pyplot as mtp


class KMeansClustering:
    def __init__(self):
        self.k = 2
        self.centroids, self.cluster1, self.cluster2 = [], [], []
        # reading file
        self.coordinates = pd.read_csv("data.csv")
        # initialize centroids
        self.initializeCentroidsAndClusters()

    def initializeCentroidsAndClusters(self):
        # Range for centroids
        X = (min(self.coordinates.X), max(self.coordinates.X))
        Y = (min(self.coordinates.Y), max(self.coordinates.Y))
        # random centroids
        for i in range(self.k):
            self.centroids.append((random.randint(X[0], X[1]), random.randint(Y[0], Y[1])))
        print("Centroids : ", self.centroids)

    def assignToCluster(self,):
        for _, coordinate in self.coordinates.iterrows():
            # calculating distance
            distance1 = ((coordinate['X'] - self.centroids[0][0])**2 + (coordinate['Y'] - self.centroids[0][1])**2)**0.5
            distance2 = ((coordinate['X'] - self.centroids[1][0]) ** 2 + (coordinate['Y'] - self.centroids[1][1]) ** 2) ** 0.5
            # assigning clusters based on distances from centroids
            if distance1 > distance2:
                self.cluster2.append((coordinate['X'], coordinate['Y']))
            else:
                self.cluster1.append((coordinate['X'], coordinate['Y']))
        # display clusters and length
        print("Cluster 1 : ", self.cluster1, "\nCluster 2 : ", self.cluster2)
        print("Size Of Cluster 1 : ", len(self.cluster1), "\nSize Of Cluster 2 : ", len(self.cluster2))

        self.recalculateCentroids()

    def recalculateCentroids(self):
        print("\nRecalculating Centroids")
        meanX, meanY, size = 0, 0, len(self.cluster1)
        # calculating means of x and y coordinates of the points in cluster
        for i in range(size):
            meanX += self.cluster1[i][0]
            meanY += self.cluster1[i][1]
        newCentroid1 = (meanX/size, meanY/size)

        meanX, meanY, size = 0, 0, len(self.cluster2)
        for i in range(len(self.cluster2)):
            meanX += self.cluster2[i][0]
            meanY += self.cluster2[i][1]
        newCentroid2 = (meanX / size, meanY / size)
        # if same centroids stop the code
        if newCentroid1 == self.centroids[0] and newCentroid2 == self.centroids[1]:
            print("Centroids : ", self.centroids, "\nCluster 1 : ", self.cluster1, "\nCluster 2 : ", self.cluster2)
            print("Size Of Cluster 1 : ", len(self.cluster1), "\nSize Of Cluster 2 : ", len(self.cluster2))
            print("Final Result!")

        else:
            # Assigning new values to centroids
            self.centroids = [newCentroid1, newCentroid2]
            print("Centroids : ", self.centroids)
            self.cluster1, self.cluster2 = [], []
            # reassign clusters based on new centroids
            print("Reassigning Clusters")
            self.assignToCluster()

    def visualizeClusters(self):
        # coordinates
        mtp.scatter(*zip(*self.cluster1), s=100, c='blue', label='Cluster 1')
        mtp.scatter(*zip(*self.cluster2), s=100, c='orange', label='Cluster 2')
        # Centroids
        mtp.scatter(self.centroids[0][0], self.centroids[0][1], s=150, c='yellow', label='Centroid 1')
        mtp.scatter(self.centroids[1][0], self.centroids[1][1], s=150, c='red', label='Centroid 2')
        # labelling
        mtp.title('K Mean Clustering')
        mtp.xlabel('X')
        mtp.ylabel('Y')
        # Display
        mtp.legend()
        mtp.show()


if __name__ == '__main__':
    Data = KMeansClustering()
    Data.assignToCluster()
    Data.visualizeClusters()
