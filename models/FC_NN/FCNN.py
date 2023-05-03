from sklearn.neural_network import MLPClassifier
import numpy as np


class FCNNClassifier(MLPClassifier):
    def __init__(self, *args, activation='relu', **kwargs):
        super().__init__(*args, activation=self.custom_activation, **kwargs)
        self.user_activation = activation

    def custom_activation(self, x):
        if self.user_activation == 'silu':
            return self.silu(x)
        elif self.user_activation == 'prelu':
            return self.prelu(x)
        elif self.user_activation == 'gelu':
            return self.gelu(x)
        elif self.user_activation == 'sigmoid':
            return self.sigmoid(x)
        else:
            return super().activation(x)

    def silu(self, x):
        return x * self.sigmoid(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def prelu(self, x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def transform(self, x):
        return softmax(self.predict(x))

    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)