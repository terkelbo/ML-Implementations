import itertools
from typing import Generator

import numpy as np
import numpy.typing as npt
from scipy.special import binom
from sklearn.base import BaseEstimator


class KernelShapExplainer:
    """
    This class implements the kernel shap specified in Theorem 2 of the paper
    """

    def __init__(self, model: BaseEstimator, nb_features: int) -> None:
        self.model = model
        self.nb_features = nb_features

        # assert that the model passed to this estimator has a method called predict
        assert hasattr(self.model, "predict")

    def _kernel_function(self, z_mark_abs: int) -> float:
        """
        Implements the kernel function from Theorem 2.
        This is denoted as pi_{x'} in the paper.

        z_mark_abs is the number of non zero features generated in that iteration
        """

        # NOTE: Handle the special case where we will divide by zero
        # this is described in the paper
        if z_mark_abs in {0, self.nb_features}:
            return 10_000_000

        # Calculate the denomination of the kernel function
        denom = (
            binom(self.nb_features, z_mark_abs)
            * z_mark_abs
            * (self.nb_features - z_mark_abs)
        )

        return (self.nb_features - 1) / denom

    def _generate_non_zero_feature_indicies(
        self,
    ) -> Generator[tuple[int, ...], None, None]:
        """
        Implements the iterator that represent the summation in Theorem 2
        It provides an iterator that yields all possible combination of
        non-zero feature combinations.
        It returns a tuple of integers where the integers represent the indicies of the
        non-zero features.
        """
        feature_indicies = list(range(self.nb_features))
        combinations = []
        for nb_features_subset in range(self.nb_features + 1):
            combinations.append(
                itertools.combinations(feature_indicies, nb_features_subset)
            )

        yield from itertools.chain.from_iterable(combinations)

    def _parametrize_summation(
        self, sample_to_explain: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        This function is helper that collects the data needed to
        parametrize the summation of Theorem 2 such that the loss function can be expressed
        as a matrix multiplication instead of a summation
        """
        # All the kernel values are collected here
        all_kernel_values = np.zeros(2**self.nb_features)

        # All the values of the h_{x}^{-1}(z') are collected here
        z_mark_inv = np.zeros((2**self.nb_features, self.nb_features))

        # All the values of z' are collected here
        # NOTE: that one more column is added to this matrix
        # this column is used to represent the bias term, i.e. phi_0
        z_mark = np.zeros((2**self.nb_features, self.nb_features + 1))
        z_mark[:, -1] = 1

        for index, non_zero_feature_indicies in enumerate(
            self._generate_non_zero_feature_indicies()
        ):
            all_kernel_values[index] = self._kernel_function(
                len(non_zero_feature_indicies)
            )
            z_mark_inv[index, non_zero_feature_indicies] = sample_to_explain[
                :, non_zero_feature_indicies
            ]
            z_mark[index, non_zero_feature_indicies] = 1

        return all_kernel_values, z_mark_inv, z_mark

    def _assert_input_shape(self, sample_to_explain: npt.NDArray) -> None:
        """
        This function asserts that the input sample to explain has the correct shape
        """
        assert sample_to_explain.shape[0] == 1
        assert sample_to_explain.shape[1] == self.nb_features

    def explain(self, sample_to_explain: npt.NDArray) -> npt.NDArray:
        """
        This function implements the kernel shap algorithm specified in Theorem 2
        """
        self._assert_input_shape(sample_to_explain)

        # Get the data needed to parametrize the summation of Theorem 2
        all_kernel_values, z_mark_inv, z_mark = self._parametrize_summation(
            sample_to_explain
        )

        # use the model predict on the z_mark_inv (i.e. f function in the paper)
        # NOTE: BaseEstimator doesn't have a predict method however it is assumed
        # that all models passed to this class will have a predict method (checked in init function)
        f_h_inv_z = self.model.predict(z_mark_inv)

        # We parametrize the solution to the loss function as a weighted least squares regression
        # the weights are the kernel values
        W = np.diag(all_kernel_values)

        # The solution to the loss function is given by the following matrix multiplication
        shap_values = np.linalg.inv(z_mark.T @ W @ z_mark) @ z_mark.T @ W @ f_h_inv_z

        return shap_values
