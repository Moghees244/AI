import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def normalize_data(data):
    normalized_data = data.copy()

    for col in data.columns:
        xmin = data[col].min()
        xmax = data[col].max()

        normalized_data[col] = (data[col] - xmin) / (xmax - xmin)

    return normalized_data


def gradient_descent(dataX, dataY, learning_rate=0.02, max_iterations=1000):
    thetas = np.ones((dataX.shape[1], 1))
    prev_cost, cost = math.inf, math.inf
    cost_per_itr = []

    for i in range(max_iterations):
        z = np.dot(dataX, thetas)
        hx = 1 / (1 + np.exp(-z))

        error = hx - dataY

        gradient = np.dot(dataX.T, error) / dataX.shape[0]
        thetas -= learning_rate * gradient

        cost = -np.mean(dataY * np.log(hx) + (1 - dataY) * np.log(1 - hx))
        if cost < prev_cost:
            cost_per_itr.append((i, cost))
            prev_cost = cost
        else:
            break

    cost_graph(cost_per_itr)
    return thetas.flatten()


def cost_graph(cost_per_itr):
    fig, ax = plt.subplots()

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Cost')

    cost_per_itr = np.array(cost_per_itr)
    ax.plot(cost_per_itr[:, 0], cost_per_itr[:, 1], linestyle='-', color='blue')

    # Show the plot
    plt.title('Change in Cost')
    plt.show()


def test(testX, testY, thetas):
    z = np.dot(testX, thetas)
    predicted_values = 1 / (1 + np.exp(-z))
    binary_predictions = (predicted_values >= 0.6).astype(int)

    correct_predictions = 0
    print("Prediction\tLabel")
    for i in range(len(binary_predictions)):
        if binary_predictions[i] == testY.iloc[i]['Label']:
            correct_predictions += 1
        print(f"{binary_predictions[i]}\t\t\t{testY.iloc[i]['Label']}")

    print("\nThe Accuracy is ", int(correct_predictions/len(binary_predictions) * 100))


if __name__ == '__main__':
    X = pd.read_csv("Regression_DataX.dat", delimiter=',', header=None, names=['Living_Area', 'Bedrooms', 'Floors'])
    Y = pd.read_csv("Regression_ClassY.dat", header=None, names=['Label'])

    # Data preprocessing
    X = normalize_data(X)
    X.insert(0, 'Bias', 1)

    # Data Splitting
    X_train, Y_train = X.iloc[:35, :], Y.iloc[:35, :]
    X_test, Y_test = X.iloc[35:, :], Y.iloc[35:, :]

    # Gradient Descent
    GD_theta = gradient_descent(X_train, Y_train)
    print("Theta = ", GD_theta)

    # Prediction
    test(X_test, Y_test, GD_theta)
