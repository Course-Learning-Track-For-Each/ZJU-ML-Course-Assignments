import numpy as np
from likelihood import likelihood


def posterior(x):
    """
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    """

    C, N = x.shape
    l = likelihood(x)
    total = np.sum(x)
    p = np.zeros((C, N))
    # TODO

    # begin answer
    class_sum = np.sum(x, axis=1)
    prior = class_sum / total
    p_x = np.zeros(N)

    for j in range(N):
        for i in range(C):
            p_x[j] += l[i, j] * prior[i]

    for i in range(C):
        for j in range(N):
            p[i, j] = l[i, j] * prior[i] / p_x[j]
    # end answer

    return p
