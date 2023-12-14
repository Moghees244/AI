import random
import pandas as pd
import matplotlib.pyplot as plt


class KMeansClustering:
    def __init__(self, k, max_itr):
        self.k = k
        self.max_iterations = max_itr
        self.iteration = 0

        self.centroids = []
        self.data = None
        self.clusters = [[] for _ in range(self.k)]

    def read_data(self):
        # reading file
        self.data = pd.read_excel("Clustering_Data.xlsx", header= None)
        print(f"Data is {self.data.shape[1]} Dimensional.\n")
        print(self.data.head(5), "\n")


    def initialize_centroids(self):
        for i in range(self.k):
            centroid = []
            # centroid for cluster
            for j in range(self.data.shape[1]):
                min_value = self.data.iloc[:, j].min()
                max_value = self.data.iloc[:, j].max()
                centroid.append(round(random.uniform(min_value, max_value), 1))

            self.centroids.append(tuple(centroid))

    def update_clusters(self):
        input(f"Press Enter For Iteration {self.iteration} : ")
        self.clusters = [[] for _ in range(self.k)]
        
        # Assign each data point to the cluster with the closest centroid
        for i in range(self.data.shape[0]):
            data_point = tuple(self.data.iloc[i, :])
            min_distance_cluster = self.get_min_distance_cluster(data_point)

            # Add the data point to the corresponding cluster
            self.clusters[min_distance_cluster].append(data_point)

        self.print_clusters()
        self.update_centroids()


    def update_centroids(self):
        new_centroids = []

        for i in range(self.k):
            new_centroid = []

            # Calculate the mean for each dimension and create a new centroid
            for dimension in range(len(self.centroids[0])):
                total_for_dimension = sum(point[dimension] for point in self.clusters[i])
                if len(self.clusters[i]) != 0:
                    mean_for_dimension = total_for_dimension / len(self.clusters[i])
                    new_centroid.append(round(mean_for_dimension, 1))
                else:
                    new_centroid.append(0)

            # Update the centroid for cluster i
            new_centroids.append(tuple(new_centroid))

        if self.centroids == new_centroids:
            print("Task Completed!")
            self.print_clusters()
            
        else:
            if self.max_iterations == self.iteration:
                print("Maximum Iteration Limit Reached.")
                return
            
            self.centroids = new_centroids
            self.iteration += 1
            
            self.update_clusters()


    def get_min_distance_cluster(self, data_point):
        distances = []
        for centroid in self.centroids:
            distance = 0
            for j in range(len(data_point)):
                # Calculate the squared difference for each dimension and add it to the sum
                distance += (data_point[j] - centroid[j]) ** 2
            distances.append(distance)

        # Find the cluster with the minimum distance
        return distances.index(min(distances))        


    def print_clusters(self):
        print("Centroids : ", self.centroids, "\n")

        for i in range(self.k):
            print(f"\nCluster {i} : ", self.clusters[i])
        print("\n\n")


    def plot_clusters(self):
        i = 0
        # Plot each cluster
        for cluster in self.clusters:
            cluster_data = list(zip(*cluster))  # Unzip the cluster data for plotting
            plt.scatter(*cluster_data, marker='o',label=f'Cluster {i}')
            i += 1

        # Plot centroids
        centroid_data = list(zip(*self.centroids))
        plt.scatter(*centroid_data, marker='x', label='Centroids')

        # Add labels and legend
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.title('K-Means Clustering')
        plt.legend()
        plt.show()        


if __name__ == "__main__":
    k = int(input("Enter The Number of Clusters : "))
    itr = int(input("Enter Maximum Number of Iterations : "))

    Kmeans = KMeansClustering(k, itr)
    Kmeans.read_data()
    Kmeans.initialize_centroids()
    Kmeans.update_clusters()
    Kmeans.plot_clusters()
