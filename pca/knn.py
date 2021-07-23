import numpy as np
import scipy.stats


def knn(x, x_train, y_train, k):
    """
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    """

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    N_test, P = x.shape
    y = np.zeros((N_test, 1))
    for i in range(N_test):
        test_vector = x[i]
        # 这里采用传统方法实现kNN，即采用最naive的方法计算测试数据和训练数据每个点的距离
        # 并从中选出最大的K个
        distance = np.linalg.norm(x_train - test_vector, axis=1)
        top_k_index = np.argsort(distance)[0: k]
        max_mode, count = scipy.stats.mode(y_train[top_k_index])
        y[i] = max_mode
    # end answer

    return y
