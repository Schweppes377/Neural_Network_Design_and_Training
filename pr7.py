import numpy as np


input_size = 1
hidden_size = 5
output_size = 1
sequence_length = 3
learning_rate = 0.1
U = np.random.randn(hidden_size, input_size)
W = np.random.randn(hidden_size, hidden_size)
V = np.random.randn(output_size, hidden_size)
bh = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)



def forward(x_seq):
    h = np.zeros((hidden_size, 1))
    for x in x_seq:
        x = np.array([[x]])
        h = sigmoid(U @ x + W @ h + bh)
    y = sigmoid(V @ h + by)
    return y, h



X = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 1]
]
Y = [
    [0],
    [1],
    [0]
]


for epoch in range(1000):
    total_loss = 0
    for x_seq, y_true in zip(X, Y):

        y_pred, h = forward(x_seq)
        y_true = np.array(y_true).reshape(-1, 1)
        loss = (y_pred - y_true)
        total_loss += np.sum(loss ** 2)
        dV = loss * dsigmoid(y_pred) @ h.T
        dby = loss * dsigmoid(y_pred)
        V -= learning_rate * dV
        by -= learning_rate * dby
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss:.4f}')


for x_seq in X:
    y_pred, _ = forward(x_seq)
    print(f'Sequence: {x_seq}, Prediction: {y_pred.ravel()[0]:.4f}')
