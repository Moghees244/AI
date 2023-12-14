import random
import math
import pandas as pd
import matplotlib.pyplot as plt


class KMedoidClustering:
    def __init__(self, k, max_itr):
        self.k = k
        self.max_iterations = max_itr
        self.iteration = 0

        self.medoids = []
        self.data = None
        self.clusters = [[] for _ in range(self.k)]

    def read_data(self):
        # reading file
        self.data = pd.read_excel("Clustering_Data.xlsx", header= None)
        print(f"Data is {self.data.shape[1]} Dimensional.\n")
        print(self.data.head(5), "\n")


    def initialize_medoids(self):
        # Shuffle the data points
        shuffled_indices = list(self.data.index)
        random.shuffle(shuffled_indices)

        # Select the first k data points as initial medoids
        for i in range(self.k):
            medoid = list(self.data.iloc[shuffled_indices[i]])
            self.medoids.append(tuple(medoid))


    def update_clusters(self):
        input(f"\nPress Enter for Iteration {self.iteration} : ")
        self.clusters = [[] for _ in range(self.k)]
        
        # Assign each data point to the cluster with the closest centroid
        for i in range(self.data.shape[0]):
            data_point = tuple(self.data.iloc[i, :])
            min_distance_cluster = self.get_min_distance_cluster(data_point)

            # Add the data point to the corresponding cluster
            self.clusters[min_distance_cluster].append(data_point)

        self.print_clusters()
        self.update_medoids()


    def update_medoids(self):
        new_medoids = []

        for i in range(self.k):
            cluster_points = self.clusters[i]

            # Choose a new medoid as the point that minimizes the sum of distances to other points in the cluster
            cost, prev_cost = math.inf, self.compute_cluster_cost(cluster_points, self.medoids[i])
            best_medoid = self.medoids[i]

            for point in cluster_points:
                cost = self.compute_cluster_cost(cluster_points, point)
                if cost < prev_cost:
                    best_medoid = point
                    prev_cost = cost

            print(f"Best Medoid for Cluster {i} is {best_medoid}. The Cost is {self.compute_cluster_cost(cluster_points, best_medoid)}")
            new_medoids.append(best_medoid)

        if self.medoids == new_medoids:
            print("Task Completed!")
            self.print_clusters()
            
        else:
            if self.max_iterations == self.iteration:
                print("Maximum Iteration Limit Reached.")
                return
            
            self.medoids = new_medoids
            self.iteration += 1
            
            self.update_clusters()

    def distance(self, point1, point2):
        return sum((a - b) ** 2 for a, b in zip(point1, point2)) ** 0.5

    def compute_cluster_cost(self, cluster_points, medoid):
        return round(sum(self.distance(point, medoid) for point in cluster_points), 2)
    
    def get_min_distance_cluster(self, data_point):
        distances = []
        for medoid in self.medoids:
            distance = 0
            for j in range(len(data_point)):
                # Calculate the squared difference for each dimension and add it to the sum
                distance += (data_point[j] - medoid[j]) ** 2
            distances.append(distance)

        # Find the cluster with the minimum distance
        return distances.index(min(distances))        


    def print_clusters(self):
        print("Medoids : ", self.medoids, "\n")

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
        centroid_data = list(zip(*self.medoids))
        plt.scatter(*centroid_data, marker='x', label='Centroids')

        # Add labels and legend
        plt.xlabel('X-axis Label')
        plt.ylabel('Y-axis Label')
        plt.title('K-Medoid Clustering')
        plt.legend()
        plt.show()        


if __name__ == "__main__":
    k = int(input("Enter The Number of Clusters : "))
    itr = int(input("Enter Maximum Number of Iterations : "))

    Kmedoid = KMedoidClustering(k, itr)
    Kmedoid.read_data()
    Kmedoid.initialize_medoids()
    Kmedoid.update_clusters()
    Kmedoid.plot_clusters()
