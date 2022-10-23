import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split

from cart.classification_tree import ClassificationTree
from cart.pruner import ClassificationTreePruner, RegressionTreePruner
from cart.regression_tree import RegressionTree
from utils.metrics import calculate_accuracy, calculate_mse


def test_pruner_on_classification_tree() -> None:  # pylint: disable=too-many-locals
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    classification_tree = ClassificationTree(
        X_train, y_train, use_exact_search=False, maximum_number_of_steps=10
    )
    classification_tree.fit()
    number_of_nodes_before = classification_tree.tree.get_total_number_of_nodes()

    # find the alpha for the tree pruner that results in the highest validation accuracy
    alphas = [0.0, 0.01, 0.05, 0.1]
    accuracies = []
    number_of_nodes = []
    for alpha in alphas:
        pruner = ClassificationTreePruner(classification_tree)
        pruned_tree = pruner.prune(alpha)

        predicted = pruned_tree.predict(X_test)
        accuracy = calculate_accuracy(predicted, y_test)
        accuracies.append(accuracy)
        number_of_nodes.append(pruned_tree.tree.get_total_number_of_nodes())

    # get the alpha with the highest accuracy
    max_accuracy = max(accuracies)
    max_accuracy_index = accuracies.index(max_accuracy)
    optimal_alpha = alphas[max_accuracy_index]

    # when alpha is zero then nothing should be pruned
    assert number_of_nodes_before == number_of_nodes[0]

    # the best alpha should be larger than zero (some pruning applied)
    assert optimal_alpha > 0

    # number of nodes should be monotonically decreasing
    assert all(
        number_of_nodes[i] >= number_of_nodes[i + 1]
        for i in range(len(number_of_nodes) - 1)
    )


def test_pruner_on_regression_tree() -> None:  # pylint: disable=too-many-locals
    X, y = fetch_california_housing(return_X_y=True)

    # we use exact search so we limit the dataset size
    # only use 500 samples
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], 500, replace=False)
    X = X[indices]
    y = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    regression_tree = RegressionTree(
        X_train, y_train, use_exact_search=False, maximum_number_of_steps=10
    )
    regression_tree.fit()
    number_of_nodes_before = regression_tree.tree.get_total_number_of_nodes()

    # find the alpha for the tree pruner that results in the highest validation mse
    alphas = [0.0, 0.01, 0.05, 0.1]
    mses = []
    number_of_nodes = []
    for alpha in alphas:
        pruner = RegressionTreePruner(regression_tree)
        pruned_tree = pruner.prune(alpha)
        predicted = pruned_tree.predict(X_test)
        mse = calculate_mse(predicted, y_test)
        mses.append(mse)
        number_of_nodes.append(pruned_tree.tree.get_total_number_of_nodes())

    # get the alpha with the lowest mse
    min_mse = min(mses)
    min_mse_index = mses.index(min_mse)
    optimal_alpha = alphas[min_mse_index]

    # when alpha is zero then nothing should be pruned
    assert number_of_nodes_before == number_of_nodes[0]

    # the best alpha should be larger than zero (some pruning applied)
    assert optimal_alpha > 0

    # number of nodes should be monotonically decreasing
    assert all(
        number_of_nodes[i] >= number_of_nodes[i + 1]
        for i in range(len(number_of_nodes) - 1)
    )
