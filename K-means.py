import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 生成模拟数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1, random_state=0)

# K-means算法
def kmeans(X, k, max_iters=100):
    # 1. 初始化质心（随机选择k个样本点作为初始质心）
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]

    # 迭代
    for _ in range(max_iters):
        # 2. 将每个样本分配给最近的质心
        labels = cdist(X, centroids).argmin(axis=1)

        # 3. 对于每个聚类，计算新的质心（均值）
        new_centroids = np.array([X[labels == i].mean(0) for i in range(k)])

        # 4. 检查质心是否变化
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, labels


# 评估聚类质量的准则函数（SSE）
def sse(X, labels, centroids):
    # 计算每个点到其质心的距离平方并求和
    distances_sq = cdist(X, centroids[labels], 'sqeuclidean')
    return distances_sq.sum()


# 运行K-means算法
k = 4  # 聚类数量
centroids, labels = kmeans(X, k)

# 计算SSE
print("SSE:", sse(X, labels, centroids))

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('K-means clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()