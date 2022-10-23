import time

from sklearn.datasets import load_iris

from cart.classification_tree import ClassificationTree
from utils.metrics import calculate_accuracy


def test_classification_tree_fits_accurately() -> None:
    """
    Tests that the classification tree can accuractely fit a dataset
    NOTE: The purpose of the test is to check that the tree can fit
    the data, hence the metrics are calculated on the training set
    """

    # load the iris dataset
    X, y = load_iris(return_X_y=True)

    # time the approximate search
    start = time.time()
    classification_tree = ClassificationTree(
        X, y, use_exact_search=False, maximum_number_of_steps=100
    )
    classification_tree.fit()
    end = time.time()
    assert end - start < 0.5

    # calculate the accuracy on the training set
    predicted = classification_tree.predict(X)
    accuracy = calculate_accuracy(predicted, y)
    assert accuracy >= 0.98

    # use the exact search
    start = time.time()
    classification_tree = ClassificationTree(X, y)
    classification_tree.fit()
    end = time.time()
    assert end - start < 0.5

    # calculate the accuracy on the training set
    predicted = classification_tree.predict(X)
    accuracy = calculate_accuracy(predicted, y)
    assert accuracy >= 0.98
