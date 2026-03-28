import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
Y = iris.target

# 对属性值进行伸缩到[-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# 对目标值进行伸缩到[0, 1]
Y_scaled = Y / (max(Y) - min(Y))

# 划分训练集和验证集
split_ratio = 0.8
split_index = int(len(X_scaled) * split_ratio)
X_train, X_val = X_scaled[:split_index].T, X_scaled[split_index:].T
Y_train, Y_val = Y_scaled[:split_index].reshape(1, -1), Y_scaled[split_index:].reshape(1, -1)

# 定义sigmoid函数和其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 初始化权重和阈值
input_size = X_train.shape[0]
hidden_size = 4
output_size = 1

np.random.seed(42)
input_weights = np.random.randn(hidden_size, input_size)
hidden_weights = np.random.randn(output_size, hidden_size)
input_bias = np.random.randn(hidden_size, 1)
hidden_bias = np.random.randn(output_size, 1)

# 训练模型
epochs = 1000
learning_rate = 0.1
a = 5  # 连续a轮的训练误差变化小于b
b = 0.0001
prev_train_loss = float('inf')
prev_val_loss = float('inf')
consecutive_loss_increase = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    total_train_loss = 0
    total_val_loss = 0

    for i in range(X_train.shape[1]):
        hidden_layer_input = np.dot(input_weights, X_train[:, i].reshape(-1, 1)) + input_bias
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_weights, hidden_layer_output) + hidden_bias
        output = sigmoid(output_layer_input)

        loss = np.square(output - Y_train[:, i])
        total_train_loss += np.sum(loss)

        output_error = Y_train[:, i] - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_error = np.dot(hidden_weights.T, output_delta)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

        hidden_weights += learning_rate * np.dot(output_delta, hidden_layer_output.T)
        hidden_bias += learning_rate * output_delta
        input_weights += learning_rate * np.dot(hidden_delta, X_train[:, i].reshape(1, -1))
        input_bias += learning_rate * hidden_delta

    avg_train_loss = total_train_loss / X_train.shape[1]
    train_losses.append(avg_train_loss)

    for i in range(X_val.shape[1]):
        hidden_layer_input = np.dot(input_weights, X_val[:, i].reshape(-1, 1)) + input_bias
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_weights, hidden_layer_output) + hidden_bias
        output = sigmoid(output_layer_input)

        loss = np.square(output - Y_val[:, i])
        total_val_loss += np.sum(loss)

    avg_val_loss = total_val_loss / X_val.shape[1]
    val_losses.append(avg_val_loss)

    if prev_train_loss - avg_train_loss < b or prev_val_loss < avg_val_loss:
        consecutive_loss_increase += 1
        if consecutive_loss_increase >= a:
            print(f"Early stopping at epoch {epoch}")
            break
    else:
        consecutive_loss_increase = 0

    prev_train_loss = avg_train_loss
    prev_val_loss = avg_val_loss

print("Training Losses:")
for idx, loss in enumerate(train_losses):
    print(f"Epoch {idx + 1}: {loss}")

print("\nValidation Losses:")
for idx, loss in enumerate(val_losses):
    print(f"Epoch {idx + 1}: {loss}")

print("\nInput Weights:")
print(input_weights)
print("\nHidden Weights:")
print(hidden_weights)
print("\nInput Bias:")
print(input_bias)
print("\nHidden Bias:")
print(hidden_bias)