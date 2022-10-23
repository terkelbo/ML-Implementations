import time

import numpy as np
from sklearn.datasets import fetch_california_housing

from cart.regression_tree import RegressionTree
from utils.metrics import calculate_mse


def test_regression_tree_fits_accurately() -> None:
    """
    Tests that the regression tree can accuractely fit a dataset
    NOTE: The purpose of the test is to check that the tree can fit
    the data, hence the metrics are calculated on the training set
    """
    # load california dataset
    X, y = fetch_california_housing(return_X_y=True)

    # time the approximate search
    start = time.time()
    regression_tree = RegressionTree(
        X, y, use_exact_search=False, maximum_number_of_steps=100
    )
    regression_tree.fit()
    end = time.time()
    assert end - start < 4

    # calculate the mse on the training set
    predicted = regression_tree.predict(X)
    mse = calculate_mse(predicted, y)
    assert mse <= 0.5

    # choose 1000 rows at random because this implementation is slow :)
    np.random.seed(42)
    random_indexes = np.random.choice(X.shape[0], 1000, replace=False)
    X_small = X[random_indexes]
    y_small = y[random_indexes]

    # time the exact search
    start = time.time()
    regression_tree = RegressionTree(X_small, y_small)
    regression_tree.fit()
    end = time.time()
    assert end - start < 4

    # calculate the mse on the training set
    predicted = regression_tree.predict(X)
    mse = calculate_mse(predicted, y)

    # NOTE: Larger MSE as the tree is fit on a subset of the data but evaluated
    # on the entire dataset
    assert mse <= 0.65
