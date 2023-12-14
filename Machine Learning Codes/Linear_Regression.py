import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# Normalizing the data using min max normalization
def normalize_data(data):
    normalized_data = data.copy()

    for col in data.columns:
        xmin = data[col].min()
        xmax = data[col].max()

        normalized_data[col] = (data[col] - xmin) / (xmax - xmin)
    return normalized_data


def gradient_descent(dataX, dataY, learning_rate=0.02, max_iterations=10000):
    thetas = np.zeros((dataX.shape[1], 1))
    prev_cost, cost = math.inf, math.inf
    cost_per_itr = []

    for i in range(max_iterations):
        hx = np.dot(dataX, thetas)
        error = hx - dataY

        gradient = np.dot(dataX.T, error) / dataX.shape[0]
        thetas -= learning_rate * gradient

        cost = np.mean(error ** 2)
        # if minima reached
        if cost < prev_cost:
            cost_per_itr.append((i, cost))
            prev_cost = cost
        else:
            break

    cost_graph(cost_per_itr)
    return thetas.flatten()


def closed_form(dataX, dataY):
    XTX = np.dot(dataX.T, dataX)
    XTX_inv = np.linalg.inv(XTX)
    XTy = np.dot(dataX.T, dataY)
    theta = np.dot(XTX_inv, XTy)

    return theta.flatten()


def cost_graph(cost_per_itr):
    fig, ax = plt.subplots()

    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Cost')

    cost_per_itr = np.array(cost_per_itr)
    ax.plot(cost_per_itr[:, 0], cost_per_itr[:, 1], linestyle='-', color='blue')

    # Show the plot
    plt.title('Change in Cost')
    plt.show()


# Predict prices using weights caluclated by algorithm
def predict(testX, testY, thetas):
    predicted_values = np.dot(testX, thetas)

    print("Predicted Price\t\tOriginal Price")
    for i in range(len(predicted_values)):
        print(f"{predicted_values[i]:.2f}\t\t\t{testY.iloc[i]['Price']}")

    scatter_plot(predicted_values, testY)


# Data representation
def scatter_plot(predicted_values, original_values):
    plt.figure(figsize=(8, 6))
    plt.scatter(original_values['Price'], predicted_values, c='b', marker='o')
    plt.xlabel("Original Price")
    plt.ylabel("Predicted Price")
    plt.title("Predicted vs. Original Prices")

    regression_line = np.polyfit(original_values['Price'], predicted_values, 1)
    y_fit = np.polyval(regression_line, original_values['Price'])
    plt.plot(original_values['Price'], y_fit, c='r', label='Linear Regression Line')

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    X = pd.read_csv("Regression_DataX.dat", delimiter=',', header=None, names=['Living_Area', 'Bedrooms', 'Floors'])
    Y = pd.read_csv("Regression_DataY.dat", header=None, names=['Price'])

    # Data preprocessing
    X = normalize_data(X)
    X.insert(0, 'Bias', 1)

    # Data Splitting
    # X_train, Y_train = X.iloc[:35, :], Y.iloc[:35, :]
    # X_test, Y_test = X.iloc[35:, :], Y.iloc[35:, :]

    # Gradient Descent
    GD_theta = gradient_descent(X, Y)
    print("Gradient Descent Theta = ", GD_theta)

    # closed form solution
    CF_theta = closed_form(X, Y)
    print("Closed Form Solution Theta = ", CF_theta)

    # Prediction
    print("\nGradient Descent Results : ")
    predict(X, Y, GD_theta)
    print("\n\nClosed Form Solution Results : ")
    predict(X, Y, CF_theta)
