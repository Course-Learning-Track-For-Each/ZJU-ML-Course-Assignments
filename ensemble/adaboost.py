import copy
import numpy as np


class Adaboost:
    """Adaboost Classifier.

    Note that this class only support binary classification.
    """

    def __init__(self,
                 base_learner,
                 n_estimator,
                 seed=2020):
        """
        Initialize the classifier.

        Args:
            base_learner: the base_learner should provide the .fit() and .predict() interface.
            n_estimator (int): The number of base learners in RandomForest.
            seed (int): random seed
        """
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]
        self._alphas = [1 for _ in range(n_estimator)]

    def fit(self, X, y):
        """
        Build the Adaboost according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        N, D = X.shape
        w = np.full(N, 1.0 / N)
        for i in range(self.n_estimator):
            dt = self._estimators[i]
            dt.fit(X, y, w)
            precision = dt.predict(X)
            error = np.sum(w[precision != y])
            self._alphas[i] = 0.5 * np.log2((1 - error) / error)
            w[precision == y] *= np.exp(-self._alphas[i])
            w[precision != y] *= np.exp(self._alphas[i])
            w /= np.sum(w)
        # end answer
        return self

    def predict(self, X):
        """
        Predict classification results for X.

        Args:
            X: testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        N = X.shape[0]
        y_pred = np.zeros(N)

        # YOUR CODE HERE
        # begin answer
        def get_distribute(y, weight):
            res = {}
            for i in range(len(y)):
                if y[i] not in res.keys():
                    res[y[i]] = 0
                res[y[i]] += weight[i]
            return res

        for i in range(N):
            predict = [dt.predict(X[i: i + 1])[0] for dt in self._estimators]
            label_weight = get_distribute(predict, self._alphas)
            y_pred[i] = max(label_weight, key=label_weight.get)
        # end answer
        return y_pred
