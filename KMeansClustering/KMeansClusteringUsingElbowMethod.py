import random
import pandas as pd
import matplotlib.pyplot as mtp


class KMeansClustering:
    def __init__(self):
        self.k = 0
        self.centroids, self.clusters, self.SSE = [], [], []
        # reading file
        self.coordinates = pd.read_csv("data.csv")

    def elbowMethod(self):
        for k in range(1, 10):
            self.k = k
            self.initializeCentroidsAndClusters()
            self.assignToCluster()
        self.plotElbow()

    def initializeCentroidsAndClusters(self):
        self.centroids = []
        # Range for centroids
        X = (min(self.coordinates.X), max(self.coordinates.X))
        Y = (min(self.coordinates.Y), max(self.coordinates.Y))
        # random centroids
        for i in range(self.k):
            self.centroids.append((random.randint(X[0], X[1]), random.randint(Y[0], Y[1])))

    def assignToCluster(self):
        for _ in range(10):
            self.clusters = [[] for i in range(self.k)]
            for _, coordinate in self.coordinates.iterrows():
                distances = []
                for centroid in self.centroids:
                    # calculating distances
                    distances.append(((coordinate['X'] - centroid[0]) ** 2 + (coordinate['Y'] - centroid[1]) ** 2))

                # assigning coordinates to clusters based on distances from centroids
                Index = distances.index(min(distances))
                self.clusters[Index].append((coordinate['X'], coordinate['Y']))

            # update centroids based on mean of coordinates in each cluster
            for i in range(self.k):
                if len(self.clusters[i]) > 0:
                    x_mean = sum([coordinate[0] for coordinate in self.clusters[i]]) / len(self.clusters[i])
                    y_mean = sum([coordinate[1] for coordinate in self.clusters[i]]) / len(self.clusters[i])
                    self.centroids[i] = (x_mean, y_mean)
        self.calculateSSE()

    def calculateSSE(self):
        sse = 0
        for i in range(self.k):
            for coordinate in self.clusters[i]:
                sse += ((coordinate[0] - self.centroids[i][0]) ** 2 + (coordinate[1] - self.centroids[i][1]) ** 2)
        self.SSE.append(sse)

    def plotElbow(self):
        mtp.plot(range(1, len(self.SSE) + 1), self.SSE, 'bx-')
        mtp.xlabel('Number of clusters (k)')
        mtp.ylabel('SSE')
        mtp.title('Elbow Method')
        mtp.show()


if __name__ == '__main__':
    Data = KMeansClustering()
    Data.elbowMethod()
