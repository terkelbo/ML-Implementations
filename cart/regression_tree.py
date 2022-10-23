import numpy as np
import numpy.typing as npt

from cart.base_tree import Node, _BaseTree


class RegressionTree(_BaseTree):
    def _predict(self, data_point: npt.NDArray) -> float:
        """
        Predicts the target value for a single data point
        """
        node = self.tree.root
        while node.split_point is not None:
            if data_point[node.split_point.feature_index] <= node.split_point.threshold:
                if node.left is None:
                    # this is a leaf node
                    prediction = self._get_y_mean(node.data_indexes)
                    break

                node = node.left
            else:
                if node.right is None:
                    # this is a leaf node
                    prediction = self._get_y_mean(node.data_indexes)
                    break

                node = node.right

        prediction = self._get_y_mean(node.data_indexes)
        assert prediction is not None, "Prediction is None in a leaf node"

        return prediction

    def _get_y_mean(self, data_indexes: npt.NDArray[np.int64]) -> float | None:
        """
        Returns the mean of the target values for the given data indexes
        Returns none if there are no data indexes
        """
        if data_indexes.shape[0] == 0:
            return None

        return np.mean(self.target[data_indexes])

    def _cost_function(
        self,
        node: Node,
        feature_index: int,
        split_point: float,
    ) -> float:
        """
        Implements the cost function from the book,
        which is MSE on both sides using the mean as the predictor
        """
        data_indexes_left, data_indexes_right = self._get_data_indexes_from_split_point(
            node, feature_index, split_point
        )
        y_mean_left = self._get_y_mean(data_indexes_left)
        y_mean_right = self._get_y_mean(data_indexes_right)
        cost_left = (
            np.sum((self.target[data_indexes_left] - y_mean_left) ** 2)
            if y_mean_left is not None
            else 0
        )
        cost_right = (
            np.sum((self.target[data_indexes_right] - y_mean_right) ** 2)
            if y_mean_right is not None
            else 0
        )
        return cost_left + cost_right
