from copy import deepcopy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from cart.base_tree import BaseTreeEstimator, Node
from cart.classification_tree import ClassificationTree


class TreePruner:
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
        tree = deepcopy(self.tree)
        sub_trees: list[BaseTreeEstimator] = []
        cost_complexity_per_tree: list[float] = []

        # add the original tree
        sub_trees.append(deepcopy(tree))
        cost_complexity_per_tree.append(self.calculate_cost_complexity(tree, alpha))
        while True:
            terminal_nodes = tree.tree.get_leaf_nodes()
            print("Number of terminal nodes:", len(terminal_nodes))

            if len(terminal_nodes) == 2:
                # we have reached the root node
                break

            min_increase = float("inf")
            node_to_prune = None
            for node in terminal_nodes:
                # calculate the cost of the parent node
                if node.parent is not None:
                    assert node.parent.split_point is not None
                    old_tree = deepcopy(tree)
                    self._remove_terminal_node_from_tree(node)
                    cost_parent = self.calculate_cost(tree)
                    tree = old_tree

                    # calculate the increase in cost
                    cost_increase = cost_parent - (self.calculate_cost(tree))
                    if cost_increase < min_increase:
                        min_increase = cost_increase
                        node_to_prune = node

            if node_to_prune is not None:
                # remove the node from the tree
                self._remove_terminal_node_from_tree(node_to_prune)

                # calculate the cost complexity of the tree
                cost_complexity = self.calculate_cost_complexity(tree, alpha)
                sub_trees.append(deepcopy(tree))
                cost_complexity_per_tree.append(cost_complexity)

        # get the tree with the lowest cost complexity
        min_cost_complexity = min(cost_complexity_per_tree)
        min_cost_complexity_index = cost_complexity_per_tree.index(min_cost_complexity)
        return sub_trees[min_cost_complexity_index]

    @staticmethod
    def calculate_cost(tree: BaseTreeEstimator) -> float:
        """
        Calculates the cost of the tree
        """
        return sum(tree.predict(tree.data) != tree.target)

    def calculate_cost_complexity(self, tree: BaseTreeEstimator, alpha: float) -> float:
        """
        Calculates the cost complexity of the tree
        """
        terminal_nodes = tree.tree.get_leaf_nodes()

        return self.calculate_cost(tree) + alpha * len(terminal_nodes)

    def _remove_terminal_node_from_tree(self, node: Node) -> None:
        """
        Removes the terminal node from the tree
        """
        parent = node.parent
        if parent is None:
            # the node is the root node
            return

        if node.is_left_node:
            parent.left = None
        else:
            parent.right = None

        # collapse the tree if the parent node only has one child
        # if the parent node only has one child then this child's parent
        # node is the parent node of the parent node
        if parent.parent is not None:
            if parent.left is None and parent.right is not None:
                new_node = parent.right
                new_node.parent = parent.parent
                parent.parent.right = new_node

            elif parent.left is not None and parent.right is None:
                new_node = parent.left
                new_node.parent = parent.parent
                parent.parent.left = new_node

    def add_terminal_node_to_tree(self, node: Node) -> None:
        """
        Adds a terminal node to the tree
        """
        parent = node.parent
        if parent is None:
            # the node is the root node
            return

        if node.is_left_node:
            parent.left = node
        else:
            parent.right = node

    def _update_data_indexes(self, node: Node) -> None:
        """
        Updates the data indexes on the subtree
        """
        if node.left is not None:
            node.left.data_indexes = node.data_indexes
            self._update_data_indexes(node.left)

        if node.right is not None:
            node.right.data_indexes = node.data_indexes
            self._update_data_indexes(node.right)


if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    classification_tree = ClassificationTree(X_train, y_train, use_exact_search=True)
    classification_tree.fit()

    # predict on the training
    predicted = classification_tree.predict(X_train)
    print("Depth before pruning:", classification_tree.tree.depth())
    pruner = TreePruner(classification_tree)
    pruned_tree = pruner.prune(0.2)
    print("Depth after pruning:", pruned_tree.tree.depth())
