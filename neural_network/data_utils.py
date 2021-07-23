import numpy as np
import os
import scipy.io as sio
import math


def get_MNIST_data():
    data = sio.loadmat('mnist_all.mat')
    X = np.concatenate([data['train{0}'.format(ix)] for ix in range(10)],
                       axis=0).reshape((-1, 1, 28, 28))
    y = np.concatenate(
        [ix * np.ones((data['train{0}'.format(ix)].shape[0]))
         for ix in range(10)], axis=0
    ).astype('uint8')
    X_test = np.concatenate([data['test{0}'.format(ix)] for ix in range(10)],
                            axis=0).reshape((-1, 1, 28, 28))
    y_test = np.concatenate(
        [ix * np.ones((data['test{0}'.format(ix)].shape[0]))
         for ix in range(10)], axis=0
    ).astype('uint8')
    p_ix = np.random.permutation(X.shape[0])
    X = X[p_ix] / 255.
    X_test = X_test / 255.
    y = y[p_ix]
    X_train, y_train = X[:50000, :], y[:50000]
    X_val, y_val = X[50000:, :], y[50000:]
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }
