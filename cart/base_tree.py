import sys
import uuid
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

sys.setrecursionlimit(10000)


@dataclass
class SplitPoint:
    cost_left: float
    cost_right: float
    cost: float
    feature_index: int
    threshold: float

    def serialise(self) -> str:
        return f"Split at feature {self.feature_index} with threshold {self.threshold} with cost {self.cost}"  # pylint: disable=line-too-long


class Node:
    def __init__(
        self,
        is_left_node: bool | None,
        data_indexes: npt.NDArray[np.int64],
        parent: "Node | None" = None,
    ):
        self.node_id = self._generate_unique_node_id()
        self.is_left_node = is_left_node
        self.data_indexes = data_indexes
        self.parent = parent
        self.left: "Node | None" = None
        self.right: "Node | None" = None
        self.split_point: SplitPoint | None = None

    def _generate_unique_node_id(self) -> str:
        """
        Generates a unique node id
        """
        return str(uuid.uuid4())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.node_id == other.node_id

    def _get_name_from_split_point(self, split_point: SplitPoint) -> str:
        """
        Returns the name of the split point based on is_left_node
        """
        if self.is_left_node:
            return f"X{split_point.feature_index} <= {split_point.threshold}"
        return f"X{split_point.feature_index} > {split_point.threshold}"

    def __repr__(self) -> str:
        """
        Prints the name if split point is given else
        prints that it is a leaf node
        """
        if self.is_leaf_node:
            return "Leaf Node"

        assert self.split_point is not None
        return self._get_name_from_split_point(self.split_point)

    @property
    def is_leaf_node(self) -> bool:
        """
        Returns true if the node is a leaf node
        """
        return self.split_point is None


class Tree:
    def __init__(self, root: Node) -> None:
        self.root = root

    def depth(self, node: Node | None = None, depth: int = 0) -> int:
        if depth == 0:
            node = self.root

        if node is None or (node.left is None and node.right is None):
            return depth

        if node.left is None:
            return self.depth(node.right, depth + 1)

        if node.right is None:
            return self.depth(node.left, depth + 1)

        return max(
            self.depth(node.left, depth + 1),
            self.depth(node.right, depth + 1),
        )

    def print(self, node: Node | None = None, depth: int = 0) -> None:
        """
        Prints the tree in a human readable format
        using the __repr__ method of the Node class
        with indentation based on the depth
        and the depth of the node
        """
        if depth == 0:
            node = self.root
        if node is None:
            return

        print(f"{'  ' * depth}Depth: {depth}, {node}")

        self.print(node.left, depth + 1)
        self.print(node.right, depth + 1)

    def get_leaf_nodes(self) -> list[Node]:
        """
        Returns a list of all the leaf nodes in the tree
        """
        leaf_nodes: list[Node] = []

        def _get_leaf_nodes(node: Node) -> None:
            if node.is_leaf_node:
                leaf_nodes.append(node)
                return

            if node.left is not None:
                _get_leaf_nodes(node.left)
            if node.right is not None:
                _get_leaf_nodes(node.right)

        _get_leaf_nodes(self.root)
        return leaf_nodes

    def get_internal_nodes(self) -> list[Node]:
        """
        Returns a list of all the internal nodes in the tree
        excluding the root node
        """
        internal_nodes: list[Node] = []

        def _get_internal_nodes(node: Node) -> None:
            if not node.is_leaf_node:
                internal_nodes.append(node)

            if node.left is not None:
                _get_internal_nodes(node.left)
            if node.right is not None:
                _get_internal_nodes(node.right)

        _get_internal_nodes(self.root)
        return internal_nodes[1:]

    def remove_subtree_after_node(self, node: Node) -> None:
        """
        Removes the subtree after the given node in place
        """
        # iterate through the nodes of the new tree
        # when we reach the node we want to remove
        # we set the left and right nodes to None
        # and return the new tree
        node_found = False

        def _remove_subtree_after_node(node: Node, new_node: Node) -> None:
            if node == new_node:
                nonlocal node_found
                node_found = True
                new_node.left = None
                new_node.right = None
                return None

            if new_node.left is not None:
                _remove_subtree_after_node(node, new_node.left)
            if new_node.right is not None:
                _remove_subtree_after_node(node, new_node.right)

            return None

        _remove_subtree_after_node(node, self.root)

        if not node_found:
            raise ValueError("Node not found in tree")

    def get_total_number_of_nodes(self) -> int:
        """
        Returns the total number of nodes in the tree
        """
        return len(self.get_leaf_nodes()) + len(self.get_internal_nodes())


class BaseTreeEstimator:
    def __init__(
        self,
        data: npt.NDArray,
        target: npt.NDArray,
        use_exact_search: bool = True,
        maximum_number_of_steps: int = 100,
        debug: bool = False,
    ) -> None:
        self.data = data
        self.target = target
        self.use_exact_search = use_exact_search
        self.debug = debug
        root_node = Node(None, np.arange(data.shape[0]))
        self.tree = Tree(root=root_node)

        # initialise a variable to hold the MSE on the training data
        self.train_mse: float | None = None
        self.step_size: npt.NDArray[np.float64] | None = None

        # we need to store the split points we have made so far
        self.split_points_already_checked: set[tuple[int, float]] = set()

        if not self.use_exact_search:
            self._set_approximate_search_step_size(maximum_number_of_steps)

    def _set_approximate_search_step_size(self, maximum_number_of_steps: int) -> None:
        """
        When using approximate search then we need a step size per feature
        to search for the split point
        """
        self.step_size = np.zeros(self.data.shape[1])
        for feature_index in range(self.data.shape[1]):
            self.step_size[feature_index] = (
                np.max(self.data[:, feature_index])
                - np.min(self.data[:, feature_index])
            ) / maximum_number_of_steps

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        """
        Predicts the target values for the given data
        """
        return np.array([self._predict(data_point) for data_point in data])

    def fit(self) -> None:
        # calculate minimum samples in node based on the size of the dataset
        minimum_samples_in_node = self.data.shape[0] // 100

        self._fit(self.tree.root, min_samples_in_node=minimum_samples_in_node)

        if self.debug:
            self.tree.print()
            print("Tree depth:", self.tree.depth())

    def _get_data_indexes_from_split_point(
        self, node: Node, feature_index: int, split_point: float
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """
        Returns the indexes of the data points that are on the
        left and right side of the split point
        """
        is_left_side = self.data[node.data_indexes, feature_index] <= split_point
        data_indexes_left = node.data_indexes[is_left_side]
        data_indexes_right = node.data_indexes[~is_left_side]

        return data_indexes_left, data_indexes_right

    def _find_split_point_exact(self, node: Node) -> SplitPoint | None:
        """
        Collects all the cost values and finds the best split point
        """
        # assert that the node has data
        assert node.data_indexes.shape[0] > 0, "Node has no data"

        all_cost_values: list[SplitPoint] = []
        for feature_index in range(self.data.shape[1]):
            # get unique split points for the feature
            # NOTE: that this also sorts the data which is needed
            split_points = np.unique(self.data[node.data_indexes, feature_index])

            # Loop through all the split points and calculate the cost
            # NOTE: we skip the last split point because it will result in no data in the right node
            for split_point in split_points[:-1]:
                # don't check a split point if it has already been checked
                if (feature_index, split_point) in self.split_points_already_checked:
                    continue

                cost_left, cost_right, cost = self._cost_function(
                    node=node, feature_index=feature_index, split_point=split_point
                )
                all_cost_values.append(
                    SplitPoint(cost_left, cost_right, cost, feature_index, split_point)
                )

        best_split_point = (
            min(all_cost_values, key=lambda x: x.cost) if all_cost_values else None
        )

        if best_split_point is not None:
            self.split_points_already_checked.add(
                (best_split_point.feature_index, best_split_point.threshold)
            )

        return best_split_point

    def _find_split_point_approximate(self, node: Node) -> SplitPoint | None:
        """
        Solves the optimisation problem using gradient descent per feature index
        It loops over all features and finds the approximation of the optimal split point
        by iterating through split points between the min/max of the feature values
        """
        # assert that the node has data
        assert node.data_indexes.shape[0] > 0, "Node has no data"

        # assert that the step size is set
        assert self.step_size is not None, "Step size is not set"

        all_cost_values: list[SplitPoint] = []
        for feature_index in range(self.data.shape[1]):
            min_value = self.data[node.data_indexes, feature_index].min()
            max_value = self.data[node.data_indexes, feature_index].max()

            # special case when the min and max are the same
            # then we can't find a split point
            if min_value == max_value:
                continue

            split_point = min_value
            step_size = self.step_size[feature_index]
            while True:
                split_point += step_size

                if split_point >= max_value:
                    break

                # don't check a split point if it has already been checked
                if (feature_index, split_point) in self.split_points_already_checked:
                    continue

                cost_left, cost_right, cost = self._cost_function(
                    node=node, feature_index=feature_index, split_point=split_point
                )
                all_cost_values.append(
                    SplitPoint(cost_left, cost_right, cost, feature_index, split_point)
                )

        best_split_point = (
            min(all_cost_values, key=lambda x: x.cost) if all_cost_values else None
        )
        if best_split_point is not None:
            self.split_points_already_checked.add(
                (best_split_point.feature_index, best_split_point.threshold)
            )

        return best_split_point

    def _fit(self, node: Node, min_samples_in_node: int) -> None:
        """
        Given a root node it grows the tree by finding the best split point
        and then recursively calling itself on the left and right nodes

        It stops when the node has less than min_samples_in_node
        """
        if len(node.data_indexes) < min_samples_in_node:
            return

        # find the best split point
        if self.use_exact_search:
            split_point = self._find_split_point_exact(node)
        else:
            split_point = self._find_split_point_approximate(node)

        if split_point is None:
            return

        # print tree depth and split point
        if self.debug:
            print("Tree depth:", self.tree.depth())
            print("Split point:", split_point.serialise())

        data_indexes_left, data_indexes_right = self._get_data_indexes_from_split_point(
            node, split_point.feature_index, split_point.threshold
        )
        node.split_point = split_point
        node.left = Node(True, data_indexes_left, parent=node)
        node.right = Node(False, data_indexes_right, parent=node)

        self._fit(node.left, min_samples_in_node)
        self._fit(node.right, min_samples_in_node)

    def _predict(self, data_point: npt.NDArray) -> float:
        raise NotImplementedError

    def get_node_prediction(self, data_indexes: npt.NDArray[np.int64]) -> float | None:
        raise NotImplementedError

    def _cost_function(
        self,
        node: Node,
        feature_index: int,
        split_point: float,
    ) -> tuple[float, float, float]:
        raise NotImplementedError
