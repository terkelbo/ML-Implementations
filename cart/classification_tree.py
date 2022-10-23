import numpy as np
import numpy.typing as npt

from cart.base_tree import Node, _BaseTree


class ClassificationTree(_BaseTree):
    def _predict(self, data_point: npt.NDArray) -> float:
        """
        Predicts the target value for a single data point
        """
        node = self.tree.root
        while node.split_point is not None:
            if data_point[node.split_point.feature_index] <= node.split_point.threshold:
                if node.left is None:
                    # this is a leaf node
                    prediction = self._get_class_majority(node.data_indexes)
                    break

                node = node.left
            else:
                if node.right is None:
                    # this is a leaf node
                    prediction = self._get_class_majority(node.data_indexes)
                    break

                node = node.right

        prediction = self._get_class_majority(node.data_indexes)
        assert prediction is not None, "Prediction is None in a leaf node"

        return prediction

    def _get_class_majority(self, data_indexes: npt.NDArray[np.int64]) -> int | None:
        """
        Returns the majority class for the given data indexes
        Returns none if there are no data indexes
        """
        if data_indexes.shape[0] == 0:
            return None

        return np.argmax(np.bincount(self.target[data_indexes]))  # type: ignore

    def _cost_function(
        self,
        node: Node,
        feature_index: int,
        split_point: float,
    ) -> float:
        """
        Implements the misclassification error cost function from the book
        """
        data_indexes_left, data_indexes_right = self._get_data_indexes_from_split_point(
            node, feature_index, split_point
        )
        class_majority_left = self._get_class_majority(data_indexes_left)
        class_majority_right = self._get_class_majority(data_indexes_right)
        cost_left = (
            np.sum(self.target[data_indexes_left] != class_majority_left)
            if class_majority_left is not None
            else 0
        )
        cost_right = (
            np.sum(self.target[data_indexes_right] != class_majority_right)
            if class_majority_right is not None
            else 0
        )
        return cost_left + cost_right
