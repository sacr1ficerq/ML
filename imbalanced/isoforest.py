from typing import List, Tuple
import numpy as np

# s(x, n) = 2 ^ {E[h(x)] / c(n)}
# c(n) = 2H(n - 1) - 2(n - 1) / n
# H(k) = Harmonic = ln(k) + 0.5772156649

debug = True

def H(k: int) -> float:
    return np.log(k) + 0.5772156649

def c_factor(n: int) -> float:
    """
    Computes average path length for an unsuccessful search in a binary search tree.
    Params:
        n: int - number of data points for BST
    """
    if n <= 1:
        return 0
    return 2*H(n - 1) - 2*(n - 1) / n

def calc_scores(E: np.ndarray, n: int) -> np.ndarray:
    return 2 ** (-E / c_factor(n))

def calc_height(X: np.ndarray, depth: int, node: 'Node') -> int | float | np.ndarray:
    """
    Calculates anomaly scores for sample in a recursive manner.
    Params:
        X: np.array - current sample, available to node
        
        depth: int - path length up to current node
        
        node: Node - current tree node
        
    Returns:
        scores: int, float or np.array - anomaly scores for sample
    """
    scores = np.zeros(X.shape[0])
    if node.kind == 'external':
        return depth + c_factor(node.size)  # single number
    elif node.kind == 'internal':
        p = X @ node.w + node.b
        less = p < 0  # left from hyperplane
        more = p >= 0  # right from hyperplane
        
        if np.any(less):
            scores[less] = calc_height(X[less], depth + 1, node.left)
        if np.any(more):
            scores[more] = calc_height(X[more], depth + 1, node.right)
        return scores
    else:
        raise ValueError

class Node(object):
    """
    A single node object for each tree. Contains information on height, current data,
    splitting hyperplane and children nodes.
    
    Attributes:
        X: np.array - data available to current node
        size: int - length of available data
        
        depth: int - depth of node

        left: Node - left child
        right: Node - right child

        kind: str - either "internal" or "external", indicates the type of current node

        w: np.array - normal vector for the splitting hyperplane
        b: float - intercept term for the splitting hyperplane
    """
    def __init__(self, X: np.ndarray, depth: int, left: 'Node | None', right: 'Node | None', kind: str, w: np.ndarray | None, b: float | None):
        """
        Node(h, left, right, kind, w, b)
        Represents the node object.
        
        Params:
            X: np.array - data available to current node
            depth: int - depth of node
            
            left: Node - left child
            right: Node - right child
            
            kind: str - either "internal" or "external", indicates the type of current node
            
            w: np.array - normal vector for the splitting hyperplane
            b: float - intercept term for the splitting hyperplane
            
        """
        self.size: int = len(X)
        
        self.depth: int = depth
        
        self.left: Node | None = left
        self.right: Node | None = right
        
        self.kind: str = kind
    
        self.w: np.ndarray | None = w
        self.b: float | None = b
    
    def __repr__(self):
        """
        For convenience only.
        """
        return f"Node(size={self.size}, depth={self.depth}, kind={self.kind})"

class RandomizedTree(object):
    """
    Single randomized tree object. Stores root and its depth (tree is built recursively).
    Attributes:
        depth: int - current tree depth
        
        max_depth: int - maximum tree depth
        
        root: Node - root node 

        internal_count: int - number of internal nodes

        external_count: int - number of external nodes
        
    """
    def __init__(self, X: np.ndarray, max_depth: int):
        """
        Single randomized tree object. Stores root and its depth (tree is built recursively).
        Params:
            X: np.array - train sample
            max_depth: int - maximum tree depth

        """
        self.depth: int = 0
        self.max_depth: int = max_depth
        
        self.internal_count: int = 0
        self.external_count: int = 0

        self.root: Node = self.grow(X, 0)
        
    def __repr__(self):
        """
        For convenience only.
        """
        
        return f"RandomizedTree(depth={self.depth}, max_depth={self.max_depth}, n_internal={self.internal_count}, n_external={self.external_count})"
        
    def grow(self, X: np.ndarray, depth: int) -> Node:
        """
        Grow tree in a recursive manner.
        Params:
            X: np.array - available train sample
            
            depth: int - current tree depth
            
        Returns:
            node: Node - a trained node with separating hyperplane data.
                         Node provides access to children if necessary (these are built recursively)
        """
        # Dont exceed self.max_depth
        # Generate random hyperplane using w and b both from in to max in all axes
        # Split X into left and right from hyperplane

        # Return Node after growing left and right Nodes

        self.depth = max(self.depth, depth)

        if depth == self.max_depth or X.shape[0] <= 1:
            self.external_count += 1
            return Node(X, depth, None, None, 'external', None, None)
        
        w = np.random.normal(0, 1, X.shape[1])
        w = w / np.linalg.norm(w)

        X_ = X @ w
        mn = np.min(X_)
        mx = np.max(X_)
        b = -np.random.uniform(mn, mx)

        less = (X_ + b) < 0
        more = (X_ + b) >= 0
        
        if not np.any(less) or not np.any(more):
            self.external_count += 1
            return Node(X, depth, None, None, 'external', None, None)

        left = self.grow(X[less], depth+1)
        right = self.grow(X[more], depth+1)
        kind = 'internal'

        self.internal_count += 1
        return Node(X, depth, left, right, kind, w, b)

    def score_samples(self, X: np.ndarray) -> np.ndarray | float:
        """
        Calculate anomaly scores for given data. You may utilize outer function `calc_height`.
        Params:
            X: np.array - data to be evaluated
            
        Returns:
            scores: np.array - estimated anomaly scores
        """
        #your code here

        # Calculate average depth of each point and return coefficient
        scores = calc_height(X, 0, self.root)
        return scores

class ExtendedIsolationForest(object):
    """
    Extended Isolation Forest object. Stores training data and trained randomized trees.
    Attributes:
        n_trees: int - number of Randomized Trees
        
        max_depth: int - maximum depth of each tree
        
        subsample_rate: float - draw `subsample_rate * X.shape[0]` samples for each tree
        
        trees: list - container for trained trees 
        
        contamination: float - estimated fraction of anomaly samples in data. Used for thresholding
        
    """
    
    def __init__(self, n_trees: int, subsample_rate: float, max_depth:int | None=None, contamination: float=0.01):
        """
        Extended Isolation Forest object. Stores training data and trained randomized trees.
        Params:
            n_trees: int - number of Randomized Trees

            subsample_rate: float - draw `subsample_rate * X.shape[0]` samples for each tree

            max_depth: int or None - maximum depth of each tree. Defaults to ceil(log_2(subsample_size)) if not provided

            contamination: float - estimated fraction of anomaly samples in data. Used for thresholding

        """
        self.n_trees: int = n_trees
        self.max_depth: int | None = max_depth
        self.subsample_rate: float = subsample_rate
        self.trees: List[RandomizedTree] = []
        self.contamination: float = contamination
        self.is_fit: bool = False
        
    def __repr__(self):
        """For convenience only."""
        
        return f"ExtendedIsolationForest(n_trees={self.n_trees}, max_depth={self.max_depth}, subsample_rate={self.subsample_rate}, contamination={self.contamination}, is_fit={self.is_fit})"
        
    def fit(self, X: np.ndarray):
        """
        Fit EIF to new data.
        Params:
            X: np.array - 2d array of samples
        """
        #your code here

        # Generate a bunch of samples
        # Create a bunch of trees and fit each tree to its sample
        N = len(X)
        n = int(self.subsample_rate * N)

        if self.max_depth is None:
            self.max_depth = np.ceil(np.log2(n))
        assert self.max_depth is not None

        for _ in range(self.n_trees):
            idx = np.random.choice(N, n, replace=False)
            x = X[idx]

            tree = RandomizedTree(x, self.max_depth)
            self.trees.append(tree)
        
        self.is_fit = True
        return self 


    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate (normalized) anomaly score for each given sample
        Params:
            X: np.array - new samples

        Returns:
            scores: np.array - anomaly scores (larger value means higher probability of a sample being an outlier)
        """
        # calculate anomaly score for each tree
        # use formula
        assert self.fit
        
        assert len(self.trees)
        assert len(X)
        h = np.zeros([len(X), len(self.trees)])

        for i, tree in enumerate(self.trees):
            h[:, i] = tree.score_samples(X)

        n = X.shape[1]
        assert n == len(self.trees[0].root.w)
        E = np.mean(h, axis=1)

        return calc_scores(E, n)

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict if given samples are outliers.
        Params:
            X: np.array - new samples

        Returns:
            labels: np.array - anomaly labels (1 for outliers, 0 for inliers)
        """
        # Score samples and round them with threshold
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        labels = (scores >= threshold).astype(int)
        return labels
