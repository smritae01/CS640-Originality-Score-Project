import os.path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.special import softmax
from os import path

# change path accordinly.
base_path = '/home/shaunak_joshi/CS640-Originality-Score-Project/'


# Custom MLPClassifier with transform method
class CustomMLPClassifier(MLPClassifier):
    def transform(self, X):
        return softmax(self.predict(X), axis=1)


data = pd.read_csv(os.path.join(base_path, 'data/combined_data.csv'), encoding="ISO-8859-1")
data.dropna(inplace=True)
train, test = train_test_split(data, test_size=0.4)

print(len(train), len(test))
print(len(data))

x_train = train.iloc[:, 2:8].values

y_train = train['Label'].values

x_test = test.iloc[:, 2:8].values

y_test = test['Label'].values

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

mlp = CustomMLPClassifier(hidden_layer_sizes=(10, 7, 6), activation='relu', solver='adam', max_iter=10000)

mlp.fit(x_train, y_train)

# RLHF part starts here
num_iterations = 5
num_episodes = 10

for iteration in range(num_iterations):
    for episode in range(num_episodes):
        # Sample a random text
        idx = np.random.choice(len(x_train))
        x_sample, y_sample = x_train[idx].reshape(1, -1), y_train[idx]

        # Get the model's prediction
        y_pred = mlp.predict(x_sample)

        # If the prediction is incorrect, update the model
        if y_pred != y_sample:
            mlp.partial_fit(x_sample, np.array([y_sample]))

# RLHF part ends here

weights = mlp.coefs_[-1]  # Access the last layer weights

abs_weights_sum = np.sum(np.abs(weights))

normalized_weights = weights / abs_weights_sum

print("Weights obtained:\n")
print("Open AI Classifier:", normalized_weights[0][0])
print("GPT Zero:(perplexity)", normalized_weights[1][0])
print("GPT Zero:(burstiness)", normalized_weights[2][0])
print("DetectGPT:(Z score)", normalized_weights[3][0])
print("GPTZero verdict", normalized_weights[4][0])
print("DetectGPT verdict", normalized_weights[5][0])
print("\n")

# np.savetxt("./data/normalized_weights.csv", normalized_weights, delimiter=",")

# df = pd.read_csv("./data/normalized_weights.csv")
# df.loc[0, ""] = 'Open AI Classifier'
# df.loc[1, ""] = 'GPT Zero: (perplexity)'
# df.loc[2, ""] = 'GPT Zero: (burstiness)'
# df.loc[3, ""] = 'DetectGPT: (Z score)'
# df.loc[4, ""] = 'GPTZero verdict'
# df.loc[5, ""] = 'DetectGPT verdict'

# df.to_csv("./data/normalized_weights.csv", index = False)

weights_file = os.path.join(base_path, 'normalized_weights_rl.csv')
df = pd.DataFrame({
    'Feature': ['Open AI Classifier', 'GPT Zero: (perplexity)', 'GPT Zero: (burstiness)', 'DetectGPT: (Z score)',
                'GPTZero verdict', 'DetectGPT verdict'],
    'Weight': [normalized_weights[0][0], normalized_weights[1][0], normalized_weights[2][0], normalized_weights[3][0],
               normalized_weights[4][0], normalized_weights[5][0]]
})
df.to_csv(weights_file, index=False)

# y_combined = np.dot(y_pred, normalized_weights)
# print("Predicted labels:\n", y_pred)

print("sum of weights:\n",
      np.abs(normalized_weights[0][0]) + np.abs(normalized_weights[1][0])
      + np.abs(normalized_weights[2][0]) + np.abs(normalized_weights[3][0]) + np.abs(normalized_weights[4][0])
      + np.abs(normalized_weights[5][0]))

y_pred = mlp.predict(x_test)

# Originality Score computation

# og_scores = np.dot(y_pred,normalized_weights)

print("Metrics on Test set:")
print("Accuracy:", accuracy_score(y_pred, y_test))
print("Precision:", precision_score(y_pred, y_test))
print("Recall:", recall_score(y_pred, y_test))
print("F1 score:", f1_score(y_pred, y_test))

with open(os.path.join(base_path, 'test_set_metrics_rl.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy_score(y_pred, y_test)}\n")
    f.write(f"Precision: {precision_score(y_pred, y_test)}\n")
    f.write(f"Recall: {recall_score(y_pred, y_test)}\n")
    f.write(f"F1 score: {f1_score(y_pred, y_test)}\n")

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.savefig('confusion_matrix.png')
plt.show()
