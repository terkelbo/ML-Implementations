import numpy as np
import numpy.typing as npt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

from shap_values.kernel_shap_explainer import KernelShapExplainer


def test_kernel_shap_explainer() -> None:
    """
    Tests the explainer on the boston dataset
    """
    X, y = fetch_california_housing(return_X_y=True)

    # make a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # create an explainer
    explainer = KernelShapExplainer(model, X.shape[1])

    # pick a sample to explain
    sample_to_explain = X[0, :].reshape(1, -1)

    # explain the sample
    shap_values = explainer.explain(sample_to_explain)

    # the only thing we can check on this dataset (as there is not feature indedependence)
    # is that the intercept is the same as the models intercept
    assert np.isclose(shap_values[-1], model.intercept_)

    # the shap values should sum to the predicted value on the sample
    assert np.isclose(shap_values.sum(), model.predict(sample_to_explain))


def make_independent_dataset() -> tuple[npt.NDArray, npt.NDArray]:
    """
    Creates a dataset where the features are independent
    Example is from the documentation of the library "shap"

    NOTE: that this dataset is perfectly independent and thus the shap values
    should be equal to the coefficients of the linear regression model
    """

    N = 2000
    X = np.zeros((N, 5))
    X[:1000, 0] = 1
    X[:500, 1] = 1
    X[1000:1500, 1] = 1
    X[:250, 2] = 1
    X[500:750, 2] = 1
    X[1000:1250, 2] = 1
    X[1500:1750, 2] = 1
    X[:, 0:3] -= 0.5
    y = 2 * X[:, 0] - 3 * X[:, 1] + 5

    # assert that the features are independent
    X_cov = np.cov(X.T)
    off_diagonal = X_cov[np.triu_indices(X_cov.shape[0], k=1)]
    assert np.allclose(off_diagonal, 0)

    return X, y


def test_independent_dataset() -> None:
    """
    Runs the explainer on a simulated dataset where the features are independent
    """

    # create a dataset where the features are independent
    X, y = make_independent_dataset()

    # make a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # create an explainer
    explainer = KernelShapExplainer(model, X.shape[1])

    # pick a sample to explain
    sample_to_explain = X[0, :].reshape(1, -1)

    # explain the sample
    shap_values = explainer.explain(sample_to_explain)

    # assuming that the dataset is indepdendent, the shap values should be equal to the coefficients
    shap_values_independent_assumption = model.coef_ * (X[0, :] - X.mean(0))
    assert np.allclose(shap_values[:-1], shap_values_independent_assumption)

    # the intercept should be the same as the models intercept
    assert np.isclose(shap_values[-1], model.intercept_)

    # the shap values should sum to the predicted value on the sample
    assert np.isclose(shap_values.sum(), model.predict(sample_to_explain))
