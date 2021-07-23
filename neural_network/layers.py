import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a mini batch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Args:
      x: (np.array) containing input data, of shape (N, d_1, ..., d_k)
      w: (np.array) weights, of shape (D, M)
      b: (np.array) biases, of shape (M,)

    Returns:
      out: output, of shape (N, M)
      cache: (x, w, b)
    """
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    # begin answer
    # 获得每个训练数据的维度
    D = 1
    for i in range(1, len(x.shape)):
        D *= x.shape[i]
    vector_x = x.reshape((x.shape[0], D))
    """ 废弃的垃圾代码
    vector_x = np.zeros((x.shape[0], D))
    for i in range(x.shape[0]):
        for l in range(x.shape[1]):
            for m in range(x.shape[2]):
                for n in range(x.shape[3]):
                    vector_x[i, 30 * l + 6 * m + n] = x[i][l][m][n]
    """
    out = np.matmul(vector_x, w) + b
    # end answer
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine layer.

    Args:
      dout: Upstream derivative, of shape (N, M)
      cache: Tuple of:
        x: Input data, of shape (N, d_1, ... d_k)
        w: Weights, of shape (D, M)

    Returns:
      dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
      dw: Gradient with respect to w, of shape (D, M)
      db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    # begin answer
    D = 1
    for i in range(1, len(x.shape)):
        D *= x.shape[i]
    vector_x = x.reshape((x.shape[0], D))
    dw = np.matmul(vector_x.T, dout)
    db = np.sum(dout, axis=0)
    dx = np.matmul(dout, w.T).reshape(x.shape)
    # end answer
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Args:
      x: Inputs, of any shape

    Returns:
      out: Output, of the same shape as x
      cache: x
    """
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    # begin answer
    cache = x
    out = x.copy()
    out[out < 0] = 0
    # end answer
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Args:
      dout: Upstream derivatives, of any shape
      cache: Input x, of same shape as dout

    Returns:
      dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    # begin answer
    dx = dout * (x >= 0)
    # end answer
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Args:
      x: Input data of shape (N, C, H, W)
      w: Filter weights of shape (F, C, HH, WW)
      b: Biases, of shape (F,)
      conv_param: A dictionary with the following keys:
        'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
        'pad': The number of pixels that will be used to zero-pad the input.

    Returns:
      out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
      cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    # begin answer
    # end answer
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Args:
      dout: Upstream derivatives.
      cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns:
      dx: Gradient with respect to x
      dw: Gradient with respect to w
      db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # begin answer
    # end answer
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max pooling layer.

    Args:
      x: Input data, of shape (N, C, H, W)
      pool_param: dictionary with the following keys:
        'pool_height': The height of each pooling region
        'pool_width': The width of each pooling region
        'stride': The distance between adjacent pooling regions

    Returns:
      out: Output data
      cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    # begin answer
    # end answer
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max pooling layer.

    Args:
      dout: Upstream derivatives
      cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
      dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    # begin answer
    # end answer
    return dx


def svm_loss(x, y):
    """Computes the loss and gradient using for multiclass SVM classification.

    Args:
      x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
      y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
      loss: Scalar giving the loss
      dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Args:
      x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
      y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
      loss: Scalar giving the loss
      dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
