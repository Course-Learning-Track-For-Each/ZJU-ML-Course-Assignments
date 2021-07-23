import numpy as np


def knn_graph(X, k, threshold):
    """
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    """

    # YOUR CODE HERE
    # begin answer
    n, p = X.shape
    W = np.zeros((n, n))
    x = np.tile(np.sum(X ** 2, axis=1), (n, 1)).T
    y = x.T
    distance = x + y - 2 * np.matmul(X, X.T)
    knn_id = np.argsort(distance)[:, 1: k + 1]
    for i in range(n):
        smaller_point_index = distance[i][knn_id[i]] <= threshold
        W[i][knn_id[i][smaller_point_index]] = 1
    return W
    # end answer
