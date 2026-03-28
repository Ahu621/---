import numpy as np

class DecisionTree:
    def __init__(self, max_features=None, max_depth=None):
        self.tree = None
        self.max_features = max_features
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.unique(y)[0]
        if X.shape[0] == 0:
            return np.bincount(y).argmax()

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None or best_threshold is None:
            return np.bincount(y).argmax()

        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > best_threshold)[0]

        if len(left_indices) == 0 or len(right_indices) == 0:
            return np.bincount(y).argmax()

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for value in values:
                left_indices = np.where(X[:, feature] <= value)[0]
                right_indices = np.where(X[:, feature] > value)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gain = self._information_gain(X, y, left_indices, right_indices)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value

        return best_feature, best_threshold

    def _information_gain(self, X, y, left_indices, right_indices):
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])

        total_samples = len(y)
        left_weight = len(left_indices) / total_samples
        right_weight = len(right_indices) / total_samples

        gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
        return gain

    def _entropy(self, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        entropy_value = 0
        total_samples = len(y)

        for count in class_counts:
            probability = count / total_samples
            entropy_value -= probability * np.log2(probability)

        return entropy_value

    def predict(self, X):
        predictions = []

        for sample in X:
            predictions.append(self._predict_sample(self.tree, sample))

        return np.array(predictions)

    def _predict_sample(self, tree, sample):
        if not isinstance(tree, dict):
            return tree

        feature = tree['feature']
        threshold = tree['threshold']
        if sample[feature] <= threshold:
            return self._predict_sample(tree['left'], sample)
        else:
            return self._predict_sample(tree['right'], sample)

class RandomForest:
    def __init__(self, n_estimators=100, max_features=None, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.tree_accuracies = []
        self.tree_predictions = []

    def fit(self, X, y):
        np.random.seed(self.random_state)

        for i in range(self.n_estimators):
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_features=self.max_features, max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            tree_predictions = tree.predict(X)
            self.tree_predictions.append(tree_predictions)
            tree_accuracy = np.mean(tree_predictions == y)
            self.tree_accuracies.append(tree_accuracy)
            print(f"Tree {i + 1} Accuracy: {tree_accuracy:.2f}")

    def predict(self, X):
        predictions = []

        for tree in self.trees:
            tree_predictions = tree.predict(X)
            predictions.append(tree_predictions)

        predictions = np.array(predictions)
        majority_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        print("Final Predictions:", majority_vote)
        return majority_vote

    def _bootstrap_sample(self, X, y):
        sample_size = int(0.15 * len(X))  # 计算采样的样本数量
        indices = np.random.choice(len(X), size=sample_size, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        return X_bootstrap, y_bootstrap

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

# Create and train the random forest model
forest = RandomForest(n_estimators=10, max_features=1, max_depth=2, random_state=5)
forest.fit(X_train, y_train)

# Predict using the trained model
predictions = forest.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# # 输出每棵树验证的预测结果
# for i, tree_pred in enumerate(forest.tree_predictions):
#     print(f"Tree {i+1} Predictions:", tree_pred)


