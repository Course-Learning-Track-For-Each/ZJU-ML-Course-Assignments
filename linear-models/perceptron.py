import numpy as np


def perceptron(X, y):
    """
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    """
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    # YOUR CODE HERE

    # begin answer
    learning_rate = 0.01
    expand_row = np.ones((N, 1))
    new_x = np.column_stack((expand_row, X.T))
    while True:
        iters += 1
        ok = 0
        for i in range(N):
            predict_class = np.matmul(new_x[i], w)
            # 表示该方案的分类是正确的
            if y[0][i] * predict_class[0] > 0:
                ok += 1
            else:
                w[:, 0] += new_x[i].T * y[0][i] * learning_rate
                learning_rate *= 0.99
        if ok == N:
            break
        # 测试非线性可分的数据集可能陷入死循环，设置一个上限及时退出止损
        # 到这里还没退出说明数据非线性导致算法失效了
        elif iters >= 200:
            break

    return w, iters
