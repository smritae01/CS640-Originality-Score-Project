import RL
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from os import path

base_path = '/home/shaunak_joshi/CS640-Originality-Score-Project/'


# Custom MLPClassifier with transform method
class CustomMLPClassifier(MLPClassifier):
    def transform(self, X):
        return softmax(self.predict(X))


# Custom activation function to normalize the output
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# Load data
data = pd.read_csv("../../data/combined_data.csv", encoding="ISO-8859-1")
data.dropna(inplace=True)
train, test = train_test_split(data, test_size=0.4)
x_train = train.iloc[:, 2:8].values
y_train = train['Label'].values
x_test = test.iloc[:, 2:8].values
y_test = test['Label'].values

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Train the MLP
mlp = CustomMLPClassifier(hidden_layer_sizes=(10,7,6), activation='relu', solver='adam', max_iter=10000)
mlp.fit(x_train, y_train)

# Create the environment
env = RL.DetectorEnvironment(x_train, y_train)

# Instantiate the PPO agent
model = PPO("MlpPolicy", env, verbose=2)

# Train the agent
model.learn(total_timesteps=10000)

# Evaluate the agent's performance
num_correct = 0
num_total = len(x_test)

for idx, x in enumerate(x_test):
    action, _ = model.predict(x, deterministic=True)
    if action == y_test[idx]:
        num_correct += 1

accuracy = num_correct / num_total
print("Accuracy:", accuracy)

# Evaluate the agent's performance on test data
accuracy = RL.evaluate(model, x_test, y_test)
print("Test Accuracy:", accuracy)
