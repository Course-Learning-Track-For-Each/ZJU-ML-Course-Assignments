import numpy as np
import kmeans


def spectral(W, k):
    """
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    """
    # YOUR CODE HERE
    # begin answer
    N = W.shape[0]
    D = np.zeros(W.shape)
    # 构造出符合条件的D矩阵
    for i in range(N):
        D[i, i] = np.sum(W[i])
    eigen_values, eigen_vectors = np.linalg.eig(D - W)
    select_k_vectors = np.argsort(eigen_values[: k])  # 取出最大的K个特征值和对应的下标
    return kmeans.kmeans(eigen_vectors[:, select_k_vectors], k)
    # end answer
