from sklearn.base import BaseEstimator
import numpy as np


def entropy(y):
    p = [len(y[y == k]) / len(y) for k in np.unique(y)]
    return -np.dot(p, np.log2(p))


def gini(y):
    p = [len(y[y == k]) / len(y) for k in np.unique(y)]
    return 1 - np.dot(p, p)


def variance(y):
    return np.var(y)


def mad_median(y):
    return np.mean(np.abs(y - np.median(y)))


criteria_dict = {
    "entropy": entropy,
    "gini": gini,
    "variance": variance,
    "mad_median": mad_median,
}


class Node:
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right


def regression_leaf(y):
    return np.mean(y)


def classification_leaf(y):
    return np.bincount(y).argmax()


class DecisionTree(BaseEstimator):
    def __init__(
        self, max_depth=np.inf, min_samples_split=2, criterion="gini", debug=False
    ):

        params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "criterion": criterion,
            "debug": debug,
        }

        for param_name, param_value in params.items():
            setattr(self, param_name, param_value)  # динамическая установка атрибутов

        super(DecisionTree, self).set_params(
            **params
        )  # устанавливаем параметры в род. класс

        self._criterion_function = criteria_dict[criterion]

        if criterion in ["variance", "mad_median"]:
            self._leaf_value = regression_leaf
        else:
            self._leaf_value = classification_leaf

        if self.debug:
            print(f"\nDecisionTree params:\n{params}")

    def _functional(self, X, y, feature_idx, threshold):
        mask = X[:, feature_idx] < threshold
        n_obj = X.shape[0]
        n_left = np.sum(mask)
        n_right = n_obj - n_left
        if n_left > 0 and n_right > 0:
            return (
                self._criterion_function(y)
                - (n_left / n_obj) * self._criterion_function(y[mask])
                - (n_right / n_obj) * self._criterion_function(y[~mask])
            )
        else:
            return 0

    # рекурсивно стоим дерево
    def _build_tree(self, X, y, depth=1):
        max_functional = 0
        best_feature_idx = None
        best_threshold = None
        n_samples, n_features = X.shape

        if len(np.unique(y)) == 1:
            return Node(labels=y)

        # елси критерий остановки не выполнен, ищем наилучшее разбиение
        if depth < self.max_depth and n_samples >= self.min_samples_split:
            if self.debug:
                print("depth = {}, n_samples = {}".format(depth, n_samples))

            for feature_idx in range(n_features):
                threshold_values = np.unique(X[:, feature_idx])  # получаем столбец
                functional_values = [
                    self._functional(X, y, feature_idx, threshold)
                    for threshold in threshold_values
                ]

                best_threshold_idx = np.nanargmax(functional_values)

                if functional_values[best_threshold_idx] > max_functional:
                    max_functional = functional_values[best_threshold_idx]
                    best_threshold = threshold_values[best_threshold_idx]
                    best_feature_idx = feature_idx
                    best_mask = X[:, feature_idx] < best_threshold

        if best_feature_idx is not None:
            if self.debug:
                print(
                    "best feature = {}, best threshold = {}".format(
                        best_feature_idx, best_threshold
                    )
                )
            return Node(
                feature_idx=best_feature_idx,
                threshold=best_threshold,
                left=self._build_tree(X[best_mask, :], y[best_mask], depth + 1),
                right=self._build_tree(X[~best_mask, :], y[~best_mask], depth + 1),
            )
        else:
            return Node(labels=y)

    def fit(self, X, y):
        if self.criterion in [
            "gini",
            "entropy",
        ]:  # при классификации запоминаем кол-во классов
            self._n_classes = len(np.unique(y))

        self.root = self._build_tree(X, y)
        return self

    def _predict_object(self, x, node=None):
        node = self.root

        while node.labels is None:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right

        return self._leaf_value(node.labels)

    def predict(self, X):
        return np.array([self._predict_object(x) for x in X])

    def _predict_proba_object(self, x, node=None):
        node = self.root

        while node.labels is None:
            if x[node.feature_idx] < node.threshold:
                node = node.left
            else:
                node = node.right

        return [
            len(node.labels[node.labels == k]) / len(node.labels)
            for k in range(self._n_classes)
        ]

    def predict_proba(self, X):
        return np.array([self._predict_proba_object(x) for x in X])
