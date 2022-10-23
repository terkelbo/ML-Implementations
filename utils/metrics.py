import numpy as np
import numpy.typing as npt


def calculate_mse(predictions: npt.NDArray, target: npt.NDArray) -> float:
    """
    Calculates the mean squared error between the predictions and target values
    """
    return np.mean((predictions - target) ** 2)


def calculate_accuracy(predictions: npt.NDArray, target: npt.NDArray) -> float:
    """
    Calculates the accuracy between the predictions and target values
    """
    return np.mean(predictions == target)
