from copy import deepcopy

import numpy as np

from cart.base_tree import BaseTreeEstimator


class BaseTreePruner:
    """
    Applies cost-complexity pruning on a fully grown tree
    starting from the terminal nodes it iteratively removes the
    terminal node that results in the smallest increase in the
    cost. The cost is calculated as the difference between the
    cost at the parent node and the sum of the costs at the
    children nodes.
    """

    def __init__(self, tree: BaseTreeEstimator):
        self.tree = tree

    def prune(self, alpha: float) -> BaseTreeEstimator:
        """
        Prunes the tree using the given alpha
        """
        sub_trees: list[BaseTreeEstimator] = []
        cost_complexity_per_tree: list[float] = []

        # add the original tree
        sub_trees.append(deepcopy(self.tree))
        cost_complexity_per_tree.append(
            self.calculate_cost_complexity(self.tree, alpha)
        )

        # loop over the internal nodes and iteratively remove the subtree from that node and down
        internal_nodes = self.tree.tree.get_internal_nodes()
        for node in internal_nodes:
            new_estimator = deepcopy(self.tree)

            # remove the subtree of the new estimator in place
            new_estimator.tree.remove_subtree_after_node(node)

            # add the new estimator to the list
            sub_trees.append(new_estimator)
            cost_complexity_per_tree.append(
                self.calculate_cost_complexity(new_estimator, alpha)
            )

        # get the tree with the lowest cost complexity
        min_cost_complexity = min(cost_complexity_per_tree)
        min_cost_complexity_index = cost_complexity_per_tree.index(min_cost_complexity)
        optimal_tree = sub_trees[min_cost_complexity_index]
        return optimal_tree

    @staticmethod
    def calculate_cost(tree: BaseTreeEstimator) -> float:
        """
        Calculates the cost of the tree
        """
        raise NotImplementedError

    def calculate_cost_complexity(self, tree: BaseTreeEstimator, alpha: float) -> float:
        """
        Calculates the cost complexity of the tree
        """
        terminal_nodes = tree.tree.get_leaf_nodes()

        return self.calculate_cost(tree) + alpha * len(terminal_nodes)


class ClassificationTreePruner(BaseTreePruner):
    """
    Applies cost-complexity pruning on a fully grown classification tree
    """

    @staticmethod
    def calculate_cost(tree: BaseTreeEstimator) -> float:
        """
        Calculates the cost of the tree (misclassification error)
        """
        y_pred = tree.predict(tree.data)
        return np.sum(y_pred != tree.target)


class RegressionTreePruner(BaseTreePruner):
    """
    Applies cost-complexity pruning on a fully grown regression tree
    """

    @staticmethod
    def calculate_cost(tree: BaseTreeEstimator) -> float:
        """
        Calculates the cost of the tree (mean squared error)
        """
        y_pred = tree.predict(tree.data)
        return np.sum((y_pred - tree.target) ** 2)
