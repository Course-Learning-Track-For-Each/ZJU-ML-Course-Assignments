import copy
import numpy as np

from collections import Counter

class RandomForest:
    """
    Random Forest Classifier.

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

    def _get_bootstrap_dataset(self, X, y):
        """
        Create a bootstrap dataset for X.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).

        Returns:
            X_bootstrap: a sampled dataset, of shape (N, D).
            y_bootstrap: the labels for sampled dataset.
        """
        # YOUR CODE HERE
        # TODO: re‐sample N examples from X with replacement
        # begin answer
        # 对训练数据进行重新采样，随机打乱原本样本集合X的顺序并返回重新生成的结果
        N, D = X.shape
        resample_index = np.random.choice(np.arange(0, N), N, replace=True)
        X_bootstrap = X[resample_index]
        y_bootstrap = y[resample_index]
        return X_bootstrap, y_bootstrap
        # end answer

    def fit(self, X, y):
        """
        Build the random forest according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        # YOUR CODE HERE
        # begin answer
        # 将数据样本用于每一个base learner的学习过程中
        for base in self._estimators:
            base.fit(X, y)
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
        # 得到每个base的预测结果
        predict_res = np.zeros((self.n_estimator, N))
        for i in range(self.n_estimator):
            predict_res[i] = self._estimators[i].predict(X)
        for i in range(N):
            number, times = Counter(predict_res.T[i]).most_common(1)[0]
            y_pred[i] = number
        # 进一步地用投票法预测出最终的结果
        # end answer
        return y_pred
