#A Multilayer perceptron to compute weights of each of the AI detection models according to the outputs they predicted

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Read the predictions and ground truth from the CSV file
data = pd.read_csv('../../data/models-output.csv')
data.dropna(inplace=True)
y_pred = data.iloc[:, :3].values

print(y_pred)
y_true = data.iloc[:, 3].values
print(y_true)

# Define an MLPRegressor to learn the weights of the models
mlp = MLPRegressor(hidden_layer_sizes=(1,), max_iter=1000)

# Fit the MLPRegressor on the predictions and ground truth
mlp.fit(y_pred, y_true)

# Get the learned weights from the MLPRegressor
weights = mlp.coefs_[0]

# Normalize the weights to sum up to 1
weights /= np.sum(weights)
print("Weights obtained:\n")
print("Open AI Classifier:", weights[0][0])
print("Zero GPT:", weights[1][0])
print("GPT Zero:", weights[2][0])
print("\n")
y_combined = np.dot(y_pred, weights)
print("Predicted labels:\n", y_combined)
# print(len(y_combined))
