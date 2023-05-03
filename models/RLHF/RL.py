import gym
import numpy as np


# Create a custom environment by extending the OpenAI Gym Env class
class DetectorEnvironment(gym.Env):
    def __init__(self, X, y):
        super(DetectorEnvironment, self).__init__()
        self.X = X
        self.y = y
        self.current_step = 0

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(6) # Assuming 6 detectors
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(6,))

    def step(self, action):
        done = False
        self.current_step += 1

        if self.current_step >= len(self.y):
            done = True

        # If we've reached the end of the dataset, reset the environment
        if self.current_step >= len(self.X):
            self.current_step = 0
            self.reset()

        state = self.X[self.current_step]
        label = self.y[self.current_step]

        reward = 0
        if action == np.argmax(state) and action == label:
            reward = 1

        return state, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.X[self.current_step]

    def render(self, mode='human'):
        pass


def evaluate(model, X, y):
    num_correct = 0
    num_total = len(X)

    for idx, x in enumerate(X):
        action, _ = model.predict(x, deterministic=True)
        if action == y[idx]:
            num_correct += 1

    accuracy = num_correct / num_total
    return accuracy