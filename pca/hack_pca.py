import numpy as np
import matplotlib.pyplot as plt
from pca import PCA


def hack_pca(filename):
    """
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    """
    # 有四个channel RGBA
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer
    # 先进行灰度图的转化
    img_gray = img_r[:, :, 0] * 0.3 + img_r[:, :, 1] * 0.59 + img_r[:, :, 2] * 0.11
    X_int = np.array((np.where(img_gray > 0)))
    X = X_int.astype(np.float64)
    # 进行主成分的分析
    p, N = X.shape
    eigen_vectors, eigen_values = PCA(X)
    print(eigen_values, eigen_vectors)
    # 转换成PCA后的图像
    Y = np.matmul(X.T, eigen_vectors).T
    Y = Y.astype(np.int32)
    p_min = np.min(Y, axis=1).reshape(p, 1)
    Y -= p_min
    bound = np.max(Y, axis=1) + 1
    res_img = np.zeros(bound)
    for i in range(Y.shape[1]):
        res_img[tuple(Y[:, i])] = img_gray[tuple(X_int[:, i])]
    res_img = res_img.T[::-1, ::-1]
    # end answer
    return res_img
