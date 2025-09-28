import numpy as np


# 激活函数及其导数
def tanh(x):
    """双曲正切激活函数"""
    return np.tanh(x)


def tanh_derivative(x):
    """tanh函数的导数"""
    return 1 - np.tanh(x) ** 2


def softmax(x):
    """softmax激活函数，用于输出层"""
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# 手动实现的RNN类
class ManualRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化RNN参数
        input_size: 输入特征维度（每个词向量的维度）
        hidden_size: 隐藏层大小
        output_size: 输出类别数
        """
        # 权重初始化
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01  # 输入到隐藏层的权重
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层到隐藏层的权重
        self.Why = np.random.randn(hidden_size, output_size) * 0.01  # 隐藏层到输出层的权重
        print(f"Wxh: {self.Wxh.shape}")
        print(f"Whh: {self.Whh.shape}")
        print(f"Why: {self.Why.shape}")

        # 偏置初始化
        self.bh = np.zeros((1, hidden_size))  # 隐藏层偏置
        self.by = np.zeros((1, output_size))  # 输出层偏置

        # 存储中间结果，用于反向传播
        self.cache = {}

    def forward(self, x):
        """
        前向传播
        x: 输入序列，形状为 (时间步长, 批量大小, 输入特征数)
        """
        # 获取输入维度信息
        seq_len, batch_size, _ = x.shape

        # 初始化隐藏状态
        h = np.zeros((batch_size, self.Whh.shape[0]))

        # 存储每个时间步的隐藏状态
        hidden_states = []

        # 遍历每个时间步
        for t in range(seq_len):
            # 当前时间步的输入
            x_t = x[t]  # 形状: (batch_size, input_size)

            # 计算隐藏状态: h_t = tanh(x_t @ Wxh + h_prev @ Whh + bh)
            h = tanh(np.dot(x_t, self.Wxh) + np.dot(h, self.Whh) + self.bh)

            # 保存当前时间步的隐藏状态
            hidden_states.append(h)

        # 用最后一个时间步的隐藏状态计算输出
        y = np.dot(h, self.Why) + self.by
        y_hat = softmax(y)  # 转换为概率分布

        # 保存中间结果用于反向传播
        self.cache['x'] = x
        self.cache['hidden_states'] = hidden_states
        self.cache['h_final'] = h
        self.cache['y'] = y
        self.cache['y_hat'] = y_hat

        return y_hat

    def backward(self, y_true):
        """
        反向传播计算梯度
        y_true: 真实标签，形状为 (batch_size, output_size)
        """
        # 获取前向传播保存的中间结果
        x = self.cache['x']
        hidden_states = self.cache['hidden_states']
        h_final = self.cache['h_final']
        y = self.cache['y']
        y_hat = self.cache['y_hat']

        seq_len, batch_size, input_size = x.shape

        # 初始化梯度
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # 输出层梯度
        dy = y_hat - y_true  # 输出误差 (batch_size, output_size)
        dWhy = np.dot(h_final.T, dy)  # (hidden_size, output_size)
        dby = np.sum(dy, axis=0, keepdims=True)  # (1, output_size)

        # 隐藏层梯度，从最后一个时间步开始反向传播
        dh_next = np.dot(dy, self.Why.T)  # 初始隐藏层误差

        # 反向遍历每个时间步
        for t in reversed(range(seq_len)):
            # 当前时间步的隐藏状态
            h = hidden_states[t]

            # 计算当前时间步的隐藏状态梯度
            dh = dh_next * tanh_derivative(h)  # (batch_size, hidden_size)

            # 如果不是第一个时间步，加上前一时间步的梯度
            if t > 0:
                h_prev = hidden_states[t - 1]
                dWhh += np.dot(h_prev.T, dh)
            else:
                # 第一个时间步的前一个隐藏状态为0
                h_prev = np.zeros_like(h)
                dWhh += np.dot(h_prev.T, dh)

            # 输入到隐藏层的权重梯度
            x_t = x[t]  # 当前时间步的输入
            dWxh += np.dot(x_t.T, dh)

            # 隐藏层偏置梯度
            dbh += np.sum(dh, axis=0, keepdims=True)

            # 计算前一时间步的隐藏状态误差
            dh_next = np.dot(dh, self.Whh.T)

        # 平均梯度（除以批量大小）
        dWxh /= batch_size
        dWhh /= batch_size
        dWhy /= batch_size
        dbh /= batch_size
        dby /= batch_size

        # 保存梯度
        self.grads = {
            'dWxh': dWxh,
            'dWhh': dWhh,
            'dWhy': dWhy,
            'dbh': dbh,
            'dby': dby
        }

    def update_parameters(self, learning_rate):
        """使用梯度下降更新参数"""
        self.Wxh -= learning_rate * self.grads['dWxh']
        self.Whh -= learning_rate * self.grads['dWhh']
        self.Why -= learning_rate * self.grads['dWhy']
        self.bh -= learning_rate * self.grads['dbh']
        self.by -= learning_rate * self.grads['dby']


# 辅助函数
def compute_loss(y_hat, y_true):
    """计算交叉熵损失"""
    batch_size = y_hat.shape[0]
    # 确保y_true是one-hot编码
    loss = -np.sum(y_true * np.log(y_hat + 1e-10)) / batch_size  # 加小值防止log(0)
    return loss


def generate_sample_data(seq_len=5, input_size=3, num_samples=100, num_classes=2):
    """生成样本数据用于测试"""
    # 输入序列: (seq_len, num_samples, input_size)
    X = np.random.randn(seq_len, num_samples, input_size)

    # 简单规则生成标签: 如果序列总和大于0则为1，否则为0
    sum_seq = np.sum(X, axis=(0, 2))
    y = np.zeros((num_samples, num_classes))
    y[sum_seq > 0, 1] = 1  # 正类
    y[sum_seq <= 0, 0] = 1  # 负类

    return X, y


# 训练函数
def train_rnn():
    # 超参数
    seq_len = 5  # 序列长度
    input_size = 3  # 输入特征维度
    hidden_size = 8  # 隐藏层大小
    output_size = 2  # 输出类别数
    learning_rate = 0.1
    epochs = 100
    batch_size = 10
    num_samples = 100  # 样本数量

    # 生成样本数据
    X, y = generate_sample_data(seq_len, input_size, num_samples, output_size)

    # 创建RNN模型
    rnn = ManualRNN(input_size, hidden_size, output_size)

    # 训练循环
    for epoch in range(epochs):
        total_loss = 0

        # 批量训练
        for i in range(0, num_samples, batch_size):
            # 获取批量数据
            batch_X = X[:, i:i + batch_size, :]
            batch_y = y[i:i + batch_size, :]

            # 前向传播
            y_hat = rnn.forward(batch_X)

            # 计算损失
            loss = compute_loss(y_hat, batch_y)
            total_loss += loss

            # 反向传播
            rnn.backward(batch_y)

            # 更新参数
            rnn.update_parameters(learning_rate)
            if i == 0 and (epoch + 1) % 50 == 0:
                print(f"epoch:{epoch+1} i == 0 y_hat: {y_hat.shape}")

        # 每10个epoch打印一次损失
        if (epoch + 1) % 10 == 0:

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (num_samples / batch_size):.4f}")

    # 测试模型
    test_X = X[:, :10, :]  # 取前10个样本作为测试
    test_y = y[:10, :]
    y_hat = rnn.forward(test_X)
    predictions = np.argmax(y_hat, axis=1)
    true_labels = np.argmax(test_y, axis=1)

    print("\n测试结果:")
    print(f"预测标签: {predictions}")
    print(f"真实标签: {true_labels}")
    print(f"准确率: {np.mean(predictions == true_labels):.2f}")


# 运行训练
if __name__ == "__main__":
    train_rnn()
