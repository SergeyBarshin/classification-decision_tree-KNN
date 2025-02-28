from decision_tree.tree import DecisionTree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def main():
    X, y = make_classification(
        n_features=2, n_redundant=0, n_samples=400, random_state=17
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=17
    )

    tree = DecisionTree(max_depth=4, debug=True)
    tree.fit(X_train, y_train)

    print("------------------")
    y_pred = tree.predict(X_test)
    print(y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    prob_pred = tree.predict_proba(X_test)
    print(prob_pred)

    if sum(np.argmax(prob_pred, axis=1) - y_pred) == 0:
        print("predict_proba works!")

    # постоим график для


if __name__ == "__main__":
    main()
