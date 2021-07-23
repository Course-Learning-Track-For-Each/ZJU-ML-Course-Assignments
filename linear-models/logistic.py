import numpy as np


def logistic(X, y):
    """
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    """
    P, N = X.shape
    w = np.zeros((P + 1, 1))

    def sigmoid(theta, x):
        return 1.0 / (1 + np.exp(-np.squeeze(np.matmul(theta.T, x))))

    X = np.concatenate((np.ones((1, N)), X), axis=0)
    y = np.array(y == 1, dtype=np.float).reshape(N)

    step = 0
    max_step = 100
    learning_rate = 0.99
    while step < max_step:
        grad = np.matmul(X, (sigmoid(w, X) - y).reshape((N, 1)))
        learning_rate *= 0.99
        w = w - learning_rate * grad
        step += 1
    return w
