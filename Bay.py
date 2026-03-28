import numpy as np
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def calculate_class_probs(self, y):
        total_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        for i, c in enumerate(classes):
            self.class_probs[c] = counts[i] / total_samples

    def calculate_feature_probs(self, X, y):
        features = X.shape[1]
        classes = np.unique(y)
        for c in classes:
            self.feature_probs[c] = {}
            for feature in range(features):
                self.feature_probs[c][feature] = {}
                unique_feature_values = np.unique(X[:, feature])
                for value in unique_feature_values:
                    self.feature_probs[c][feature][value] = (np.sum((X[:, feature] == value) & (y == c)) + 1) / (
                                np.sum(y == c) + len(unique_feature_values))

    def predict(self, X):
        predictions = []
        for x in X:
            max_prob = -1
            max_class = None
            for c in self.class_probs.keys():
                class_prob = np.log(self.class_probs[c])
                for i, feature in enumerate(x):
                    if feature in self.feature_probs[c][i]:
                        class_prob += np.log(self.feature_probs[c][i][feature])
                if class_prob > max_prob or max_class is None:
                    max_prob = class_prob
                    max_class = c
            predictions.append(max_class)
        return predictions


# 加载鸢尾花数据集
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练朴素贝叶斯分类器
nb = NaiveBayesClassifier()
nb.calculate_class_probs(y_train)
nb.calculate_feature_probs(X_train, y_train)

# 预测
predictions = nb.predict(X_test)

# 输出验证集的输入项和对应的标签和类别
print("验证集数据和标签:")
print("花萼长度、花萼宽度、花瓣长度、花瓣宽度")
for i, x in enumerate(X_test):
    label = y_test[i]
    class_name = iris.target_names[label]
    pred_class_name = iris.target_names[predictions[i]]
    print(f"特征值: {x}, 标签: {label}, 真实类别: {class_name}, 预测类别: {pred_class_name}")

# 计算准确率
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"\n准确率: {accuracy}")