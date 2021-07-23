import numpy as np


def linear_regression(X, y):
    """
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    """
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    # 原本的每个样本x是按列存放的，这里需要对矩阵进行转置并在开头加上一列1，成为N*(P+1)维的矩阵
    new_x = np.column_stack((np.ones((N, 1)), X.T))
    part1 = np.linalg.inv(np.matmul(new_x.T, new_x))
    part2 = np.matmul(new_x.T, y.T)
    w = np.matmul(part1, part2)
    # end answer
    return w
