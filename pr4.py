import numpy as np
import pandas as pd

date = {
    "x1": [1, 1, 0, 0],
    "x2": [1, 0, 1, 0],
    "y": [1, 0, 0, 0]
}
df = pd.DataFrame(date)
X_train = df[['x1', 'x2']].values
Y_train = df['y'].values
centers = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
sigma = 1.0

def rbf_activation(x, center, sigma):

    distance = np.linalg.norm(x - center)
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

F = np.zeros((len(X_train), len(centers)))
for i, x in enumerate(X_train):
    for j, c in enumerate(centers):
        F[i, j] = rbf_activation(x, c, sigma)


weights = np.linalg.pinv(F) @ Y_train

for x in X_train:
    hidden_activations = np.array([rbf_activation(x, c, sigma) for c in centers])
    output = hidden_activations @ weights

    print((x[0], x[1]), str(output))