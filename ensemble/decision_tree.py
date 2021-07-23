import numpy as np


class DecisionTree:
    """
    Decision Tree Classifier.

    Note that this class only supports binary classification.
    """

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 first_cut=False,
                 last_cut=False,
                 sample_feature=False):
        """Initialize the classifier.

        Args:
            criterion (str): the criterion used to select features and split nodes.
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between under fitting and over fitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between under fitting and over fitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        """
        if criterion == 'infogain_ratio':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be infogain_ratio or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature
        self.first_cut = first_cut
        self.last_cut = last_cut

    def fit(self, X, y, sample_weights=None):
        """
        Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
            sample_weights: weights for each samples, of shape (N,).
        """
        if sample_weights is None:
            # if the sample weights is not provided, then by default all
            # the samples have unit weights.
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)[0]
        return self

    @staticmethod
    def entropy(y, sample_weights):
        """Calculate the entropy for label.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the entropy for y.
        """
        entropy = 0.0
        # begin answer
        weights_sum = np.sum(sample_weights)
        # 统计各个标签在样本中出现的概率
        probability = DecisionTree.get_label_distribute(y, sample_weights)
        # 计算样本的信息熵
        for p in probability.values():
            entropy -= (p / weights_sum) * np.log2(p / weights_sum)
        # end answer
        return entropy

    def _information_gain(self, X, y, index, sample_weights):
        """
        Calculate the information gain given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain calculated.
        """

        # YOUR CODE HERE
        # begin answer
        info_gain = self.entropy(y, sample_weights)
        N, D = X.shape
        # 得到选定特征的取值分布表以及所有取这个值的样本的下标
        weight_sum = np.sum(sample_weights)
        """
        val_state = {}
        for i in range(N):
            if X[i, index] not in val_state.keys():
                val_state[X[i, index]] = []
            val_state[X[i, index]].append(i)
        for val in val_state.keys():
            select_index = val_state[val]
            entropy_v = self.entropy(y[select_index], sample_weights[select_index])
            info_gain -= np.sum(sample_weights[select_index]) * entropy_v / weight_sum
        """
        for val in np.unique(X.T[index]):
            chosen_index = X.T[index] == val
            probability = np.sum(sample_weights[chosen_index]) / weight_sum
            info_gain -= probability * self.entropy(y[chosen_index], sample_weights[chosen_index])
        # end answer
        return info_gain

    def _information_gain_ratio(self, X, y, index, sample_weights):
        """
        Calculate the information gain ratio given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain ratio calculated.
        """
        # YOUR CODE HERE
        # begin answer
        """
        info_gain_ratio = self._information_gain(X, y, index, sample_weights)
        val_state = {}
        for i in range(X.shape[0]):
            if X[i, index] not in val_state.keys():
                val_state[X[i, index]] = []
            val_state[X[i, index]].append(i)
        sample_sum = np.sum(sample_weights)
        for val in val_state.keys():
            select_index = val_state[val]
            select_ratio = np.sum(sample_weights[select_index]) / sample_sum
            split_information -= select_ratio * np.log2(select_ratio)
        info_gain_ratio /= split_information
        """
        entropy_sum = self.entropy(X.T[index], sample_weights)
        if entropy_sum == 0:
            entropy_sum = 1e-7
        info_gain_ratio = self._information_gain(X, y, index, sample_weights) / entropy_sum
        # end answer
        return info_gain_ratio

    @staticmethod
    def get_label_distribute(y, weight):
        distribute, N = {}, y.shape[0]
        for i in range(N):
            if y[i] not in distribute.keys():
                distribute[y[i]] = 0
            distribute[y[i]] += weight[i]
        return distribute

    @staticmethod
    def gini_impurity(y, sample_weights):
        """
        Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """
        gini = 1
        # YOUR CODE HERE
        # begin answer
        probability = DecisionTree.get_label_distribute(y, sample_weights)
        total_weight = np.sum(sample_weights)
        gini -= np.sum((np.array(list(probability.values())) / total_weight) ** 2)
        # end answer
        return gini

    def _gini_purification(self, X, y, index, sample_weights):
        """
        Calculate the resulted gini impurity given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the resulted gini impurity after splitting by this feature.
        """
        new_impurity = 1
        # YOUR CODE HERE
        # begin answer
        for value in np.unique(X.T[index]):
            chosen_data = X.T[index] == value
            probability = np.sum(sample_weights[chosen_data])
            new_impurity -= probability * self.gini_impurity(
                y[chosen_data], sample_weights[chosen_data])
        # end answer
        return new_impurity

    def _split_dataset(self, X, y, index, value, sample_weights, X_valid=None, y_valid=None):
        """
        Return the split of data whose index-th feature equals value.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for splitting.
            value: the value of the index-th feature for splitting.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (np.array): the subset of X whose index-th feature equals value.
            (np.array): the subset of y whose index-th feature equals value.
            (np.array): the subset of sample weights whose index-th feature equals value.
        """
        sub_X, sub_y, sub_sample_weights = X, y, sample_weights
        # YOUR CODE HERE
        # Hint: Do not forget to remove the index-th feature from X.
        # begin answer
        chosen_data = X.T[index] == value
        sub_X = np.delete(X[chosen_data], index, axis=1)
        sub_y = y[chosen_data]
        sub_sample_weights = sample_weights[chosen_data]
        # end answer
        if not self.first_cut and not self.last_cut:
            return sub_X, sub_y, sub_sample_weights
        else:
            sub_chosen_data = X_valid[:, index] == value
            sub_X_valid = np.delete(X_valid[sub_chosen_data], index, axis=1)
            sub_y_valid = y_valid[sub_chosen_data]
            return sub_X, sub_y, sub_sample_weights, sub_X_valid, sub_y_valid

    def _choose_best_feature(self, X, y, sample_weights):
        """
        Choose the best feature to split according to criterion.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the index for the best feature
        """
        best_feature_idx = 0
        # YOUR CODE HERE
        # Note that you need to implement the sampling feature part here for random forest!
        # Hint: You may find `np.random.choice` is useful for sampling.
        # begin answer
        N, D = X.shape
        if self.sample_feature:
            m = max(int(D ** 0.5), 1)
            new_index = np.random.choice(range(D), m, replace=False)
            # 根据判断标准选择出最合适的划分特征
            best_feature_idx = max([(index, self.criterion(X, y, index, sample_weights))
                                    for index in new_index], key=lambda x: x[1])[0]
        else:
            best_feature_idx = max([(index, self.criterion(X, y, index, sample_weights))
                                    for index in range(D)], key=lambda x: x[1])[0]
        # end answer
        return best_feature_idx

    @staticmethod
    def majority_vote(y, sample_weights=None):
        """
        Return the label which appears the most in y.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        majority_label = y[0]
        # YOUR CODE HERE
        # begin answer
        # 统计样本的分布并获得出现概率最高的样本
        probability = DecisionTree.get_label_distribute(y, sample_weights)
        majority_label = max(probability, key=probability.get)
        # end answer
        return majority_label

    def _build_tree(self, X, y, feature_names, depth, sample_weights, X_valid=None, y_valid=None):
        """
        Build the decision tree according to the data.

        Args:
            X: (np.array) training features, of shape (N, D).
            y: (np.array) vector of training labels, of shape (N,).
            feature_names (list): record the name of features in X in the original dataset.
            depth (int): current depth for this node.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (dict): a dict denoting the decision tree.
            Example:
                The first best feature name is 'title', and it has 5 different values:
                0,1,2,3,4. For 'title' == 4, the next best feature name is 'pclass',
                we continue split the remain data. If it comes to the leaf, we use
                the majority_label by calling majority_vote.
                my_tree = {
                    'title': {
                        0: subtree0,
                        1: subtree1,
                        2: subtree2,
                        3: subtree3,
                        4: {
                            'pclass': {
                                1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                                2: majority_vote([1, 0, 1, 1]) # which is 1
                                3: majority_vote([0, 0, 0]) # which is 0
                            }
                        }
                    }
                }
        """
        mytree = dict()
        # YOUR CODE HERE
        # TODO: Use `_choose_best_feature` to find the best feature to split the X.
        #  Then use `_split_dataset` to get subtrees.
        # Hint: You may find `np.unique` is useful. build_tree is recursive.
        # begin answer
        directly_estimate = self.majority_vote(y, sample_weights)
        number = np.sum(y_valid == directly_estimate)
        if len(set(y)) == 1 or self._all_the_same(X):
            return directly_estimate, number
        if X.shape[0] < self.min_samples_leaf or depth > self.max_depth:
            return directly_estimate, number
        chosen_index = self._choose_best_feature(X, y, sample_weights)
        mytree[feature_names[chosen_index]] = {}
        new_feature_names = feature_names.copy()
        del new_feature_names[chosen_index]
        if self.first_cut:
            new_number = 0
            for branch_value in np.unique(X[:, chosen_index]):
                sub_X, sub_y, sub_sample_weights, sub_X_valid, sub_y_valid = \
                    self._split_dataset(X, y, chosen_index, branch_value,
                                        sample_weights, X_valid, y_valid)
                new_number += np.sum(sub_y_valid == self.majority_vote(sub_y))
            if number >= new_number:
                return directly_estimate, number
        new_number = 0
        for branch_value in np.unique(X[:, chosen_index]):
            if not self.first_cut and not self.last_cut:
                sub_X, sub_y, sub_sample_weights = \
                    self._split_dataset(X, y, chosen_index, branch_value, sample_weights)
                mytree[feature_names[chosen_index]][branch_value], ok = \
                    self._build_tree(sub_X, sub_y, new_feature_names, depth + 1, sub_sample_weights)
            else:
                sub_X, sub_y, sub_sample_weights, sub_X_valid, sub_y_valid = \
                    self._split_dataset(X, y, chosen_index, branch_value,
                                        sample_weights, X_valid, y_valid)
                mytree[feature_names[chosen_index]][branch_value], ok = \
                    self._build_tree(sub_X, sub_y, new_feature_names, depth + 1,
                                     sub_sample_weights, sub_X_valid, sub_y_valid)
            new_number += ok
        if self.last_cut:
            if number > new_number:
                return directly_estimate, number
        # end answer
        return mytree, new_number

    def _all_the_same(self, X):
        for i in range(1, X.shape[0]):
            if not (X[0] == X[i]).all():
                return False
        return True

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: (pd.Dataframe) testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree, x):
            """
            Classify a single sample with the fitted decision tree.

            Args:
                x: ((pd.Dataframe) a single sample features, of shape (D,).

            Returns:
                (int): predicted testing sample label.
            """
            # YOUR CODE HERE
            # begin answer
            if type(tree) == dict:
                feature_name, content = list(tree.items())[0]
                current_value = x[feature_name].values[0]
                if current_value in content:
                    return _classify(content[current_value], x)
                else:
                    import random
                    return _classify(content[random.choice(list(content.keys()))], x)
            else:
                return tree
            # end answer

        # YOUR CODE HERE
        # begin answer
        predict = []
        for i in range(X.shape[0]):
            predict.append(_classify(self._tree, X[i: i + 1]))
        return np.array(predict)
        # end answer

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
