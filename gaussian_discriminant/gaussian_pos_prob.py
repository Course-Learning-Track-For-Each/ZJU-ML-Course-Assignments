import numpy as np


def gaussian_pos_prob(X, Mu, Sigma, Phi):
    """
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    """
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    # 先计算likelihood
    likelihood = np.zeros((N, K))
    for i in range(N):
        p_x = 0
        for j in range(K):
            x_minus_mu = X[:, i] - Mu[:, j]
            sigma = Sigma[:, :, j]
            det_sigma = np.linalg.det(sigma)
            inv_sigma = np.linalg.inv(sigma)
            base = 1.0 / (2 * np.pi * np.sqrt(np.abs(det_sigma)))
            exponent = np.matmul(np.matmul(x_minus_mu.T, inv_sigma), x_minus_mu) * -0.5
            likelihood[i, j] = base * np.exp(exponent)
            p_x += likelihood[i, j] * Phi[j]
        for j in range(K):
            p[i, j] = likelihood[i, j] * Phi[j] / p_x
    return p
