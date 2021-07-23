import numpy as np
from scipy import linalg


def ridge(X, y, lmbda):
    """
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    """
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    I = np.identity(P + 1)
    I[0][0] = 0
    new_x = np.concatenate((np.ones((1, N)), X), axis=0)
    part1 = np.matmul(new_x, new_x.T) + lmbda * I
    part2 = np.matmul(new_x, y.T)
    w = np.matmul(linalg.pinv(part1), part2)
    # end answer
    return w
