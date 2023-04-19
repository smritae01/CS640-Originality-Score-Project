import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# TODO implement better activation function with custom MLP regressor class with embedded forward pass
# TODO update model with the updated dataset columns

# Custom MLPRegressor with transform method
class CustomMLPRegressor(MLPRegressor):
    def transform(self, X):
        return softmax(self.predict(X))

# Custom activation function to normalize the output
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

data = pd.read_csv('/home/ssj/CS640-Originality-Score-Project/data/AI_data.csv')
data = data.drop(['Unnamed: 8', 'DetectGPT', 'GPT Zero '], axis=1)
data.dropna(inplace=True)

# data.head(10)

data['Label'] = 1
y_pred = data.iloc[:, 2:6].values
# print(y_pred)
y_true = data['Label'].values

scaler = StandardScaler()
y_pred = scaler.fit_transform(y_pred)

mlp = CustomMLPRegressor(hidden_layer_sizes=(10,7,4), activation='relu', solver='adam', max_iter=10000)

mlp.fit(y_pred, y_true)

weights = mlp.coefs_[-1]  # Access the last layer weights

abs_weights_sum = np.sum(np.abs(weights))

normalized_weights = weights / abs_weights_sum

print("Weights obtained:\n")
print("Open AI Classifier:", normalized_weights[0][0])
print("GPT Zero:(perplexity)", normalized_weights[1][0])
print("GPT Zero:(burstiness)", normalized_weights[2][0])
print("DetectGPT:(Z score)", normalized_weights[3][0])
print("\n")

# y_combined = np.dot(y_pred, normalized_weights)
# print("Predicted labels:\n", y_pred)

print("sum of weights:\n", 
      np.abs(normalized_weights[0][0])+np.abs(normalized_weights[1][0])+
      np.abs(normalized_weights[2][0])+np.abs(normalized_weights[3][0]))
