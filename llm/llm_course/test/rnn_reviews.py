import numpy as np
import jieba  # 用于中文分词
import pandas as pd


# 激活函数及其导数
def tanh(x):
    """双曲正切激活函数"""
    return np.tanh(x)


def tanh_derivative(h):
    """h 已经是激活后的值"""
    return 1 - h ** 2


def softmax(x):
    """softmax激活函数，用于输出层"""
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))  # 减最大值防止溢出
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# 手动实现的RNN类 - 保持原有细节不变
class ManualRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化RNN参数
        input_size: 输入特征维度（每个词向量的维度）
        hidden_size: 隐藏层大小
        output_size: 输出类别数
        """
        # 权重初始化（用小随机数，避免梯度爆炸）
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

        # 初始化隐藏状态（每个样本的初始隐藏状态为0）
        h = np.zeros((batch_size, self.Whh.shape[0]))

        # 存储每个时间步的隐藏状态
        hidden_states = []

        # 遍历每个时间步
        for t in range(seq_len):
            # 当前时间步的输入（取第t个时间步的所有样本）
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

        # 输出层梯度（交叉熵损失对y的导数）
        dy = y_hat - y_true  # 输出误差 (batch_size, output_size)
        dWhy = np.dot(h_final.T, dy)  # (hidden_size, output_size)
        dby = np.sum(dy, axis=0, keepdims=True)  # (1, output_size)

        # 隐藏层梯度，从最后一个时间步开始反向传播
        dh_next = np.dot(dy, self.Why.T)  # 初始隐藏层误差（输出层对最后一个h的导数）

        # 反向遍历每个时间步
        for t in reversed(range(seq_len)):
            # 当前时间步的隐藏状态
            h = hidden_states[t]

            # 计算当前时间步的隐藏状态梯度（tanh导数 * 后续误差）
            dh = dh_next * tanh_derivative(h)  # (batch_size, hidden_size)

            # 计算隐藏层到隐藏层的权重梯度（前一个h对当前h的贡献）
            if t > 0:
                h_prev = hidden_states[t - 1]  # 前一个时间步的h
            else:
                h_prev = np.zeros_like(h)  # 第一个时间步没有前h，用0
            dWhh += np.dot(h_prev.T, dh)

            # 计算输入到隐藏层的权重梯度（当前x对h的贡献）
            x_t = x[t]  # 当前时间步的输入
            dWxh += np.dot(x_t.T, dh)

            # 隐藏层偏置梯度（偏置对所有样本的贡献求和）
            dbh += np.sum(dh, axis=0, keepdims=True)

            # 更新下一个（前一个时间步）的隐藏层误差
            dh_next = np.dot(dh, self.Whh.T)

        # 平均梯度（除以批量大小，避免批量大小影响梯度尺度）
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
    """计算交叉熵损失（避免log(0)，加小值1e-10）"""
    batch_size = y_hat.shape[0]
    loss = -np.sum(y_true * np.log(y_hat + 1e-10)) / batch_size  # 平均到每个样本
    return loss


# 中文文本处理函数（重点修复词汇表构建）
def load_chinese_data(file_path):
    """加载数据：读取CSV，获取文本和标签"""
    df = pd.read_csv(file_path, header=0)  # header=0 表示第一行为列名
    # 处理可能的列名不一致（确保取到正确的文本和标签列）
    if 'review' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV文件必须包含 'review'（文本列）和 'label'（标签列）")

    texts = df['review'].tolist()  # 文本列表
    labels = df['label'].tolist()  # 标签列表（0/1）

    # 处理标签：确保是整数
    labels = [int(label) for label in labels if str(label).isdigit()]
    # 同步过滤文本（避免标签过滤后文本数量不匹配）
    texts = [text for text, label in zip(texts, df['label'].tolist()) if str(label).isdigit()]

    # 打乱数据（保持文本和标签的对应关系）
    indices = list(range(len(texts)))
    np.random.shuffle(indices)
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    print(f"加载完成：共 {len(texts)} 条数据")
    return texts, labels


def preprocess_texts(texts, max_seq_len=10, embedding_dim=5):
    """
    预处理中文文本：分词 → 构建词汇表 → 文本转序列向量
    返回：X（输入序列，shape=(max_seq_len, num_samples, embedding_dim)）、word_to_idx（词→索引）、word_vectors（词向量）
    """
    # -------------------------- 第一步：对每个文本分词，得到“文本-词列表”映射 --------------------------
    tokenized_texts = []  # 存储每个文本的词列表（len(tokenized_texts) = len(texts)）
    for text in texts:
        # 处理NaN和非字符串类型（转为空字符串）
        if isinstance(text, float) and np.isnan(text):
            text = ""
        else:
            text = str(text).strip()

        # 中文分词（用jieba的精确模式）
        words = jieba.lcut(text)  # lcut返回列表，比cut更方便
        # 过滤空词（避免分词后出现空字符串）
        words = [word for word in words if word.strip()]
        tokenized_texts.append(words)

    # 调试：打印前3个文本的分词结果，验证是否正确
    print("\n前3个文本的分词结果：")
    for i in range(min(3, len(tokenized_texts))):
        print(f"文本{i + 1}：{tokenized_texts[i]}")

    # -------------------------- 第二步：构建词汇表（word_to_idx） --------------------------
    word_to_idx = {}  # 词 → 唯一索引（从0开始）
    idx = 0
    for words in tokenized_texts:  # 遍历每个文本的词列表
        for word in words:  # 遍历每个词
            if word not in word_to_idx:  # 只添加未出现过的词
                word_to_idx[word] = idx
                idx += 1

    vocab_size = len(word_to_idx)
    print(f"\n词汇表构建完成：共 {vocab_size} 个不重复词")
    # 调试：打印前5个词的索引映射
    print("前5个词的索引映射：", dict(list(word_to_idx.items())[:5]))

    # -------------------------- 第三步：生成随机词向量（实际用预训练向量更好） --------------------------
    # 词向量形状：(词汇表大小, 词向量维度)
    word_vectors = np.random.randn(vocab_size, embedding_dim) * 0.01  # 小随机数初始化

    # -------------------------- 第四步：文本转序列向量（固定长度max_seq_len） --------------------------
    num_samples = len(texts)  # 样本数量（即文本数量）
    # 输入序列形状：(时间步长, 批量大小, 输入特征数) → (max_seq_len, num_samples, embedding_dim)
    X = np.zeros((max_seq_len, num_samples, embedding_dim))

    for i in range(num_samples):  # 遍历每个样本（每个文本）
        words = tokenized_texts[i]  # 当前文本的词列表
        # 遍历每个词，填充到序列中（不足max_seq_len补0，超过则截断）
        for t in range(min(len(words), max_seq_len)):
            word = words[t]
            if word in word_to_idx:  # 只处理在词汇表中的词
                word_idx = word_to_idx[word]
                X[t, i, :] = word_vectors[word_idx]  # 第t个时间步、第i个样本的词向量

    # 调试：打印输入序列的形状，验证是否正确
    print(f"\n输入序列X的形状：{X.shape}")  # 应输出 (max_seq_len, num_samples, embedding_dim)
    return X, word_to_idx, word_vectors


def convert_labels_to_onehot(labels):
    """将标签转换为one-hot编码（适用于二分类）"""
    num_classes = 2  # 0:负面，1:正面
    num_samples = len(labels)
    y_onehot = np.zeros((num_samples, num_classes))
    for i, label in enumerate(labels):
        if label in [0, 1]:  # 确保标签是0或1
            y_onehot[i, label] = 1
    return y_onehot


# 训练函数
def train_rnn():
    # 超参数（可根据数据调整）
    max_seq_len = 10  # 每个文本的最大词数（时间步长）
    embedding_dim = 5  # 词向量维度（不宜过大，避免计算量增加）
    hidden_size = 8  # 隐藏层大小（小数据集用小隐藏层，避免过拟合）
    output_size = 2  # 输出类别数（二分类：0负面，1正面）
    learning_rate = 0.01  # 学习率（过小收敛慢，过大易震荡）
    epochs = 100  # 训练轮次
    batch_size = 10  # 批量大小（小数据集用小批量）

    # 1. 加载并预处理数据
    try:
        texts, labels = load_chinese_data(
            R'E:\学习\ai\慕课llm算法特训002.LLM大语音模型算法特训 带你转型AI大语音模型算法工程师\源码+PDF课件\源码\llm\llm_course\test\ChnSentiCorp_htl_all.csv'
        )
    except Exception as e:
        print(f"数据加载失败：{e}")
        return

    # 文本转序列向量
    X, word_to_idx, word_vectors = preprocess_texts(texts, max_seq_len, embedding_dim)
    # 标签转one-hot编码
    y = convert_labels_to_onehot(labels)
    num_samples = X.shape[1]  # 样本数量（X的第2个维度是样本数）

    # 2. 创建RNN模型
    rnn = ManualRNN(input_size=embedding_dim, hidden_size=hidden_size, output_size=output_size)

    # 3. 训练循环
    print("\n开始训练RNN...")
    for epoch in range(epochs):
        total_loss = 0.0
        # 批量遍历数据（从0到num_samples，步长为batch_size）
        for i in range(0, num_samples, batch_size):
            # 截取当前批量的数据（注意避免超出样本数）
            end_idx = min(i + batch_size, num_samples)
            batch_X = X[:, i:end_idx, :]  # 批量输入：(max_seq_len, batch_size, embedding_dim)
            batch_y = y[i:end_idx, :]  # 批量标签：(batch_size, output_size)

            # 前向传播：计算预测概率
            y_hat = rnn.forward(batch_X)
            # 计算损失
            loss = compute_loss(y_hat, batch_y)
            total_loss += loss * (end_idx - i)  # 乘以批量大小，最后再平均

            # 反向传播：计算梯度
            rnn.backward(batch_y)
            # 更新参数
            rnn.update_parameters(learning_rate)

        # 计算平均损失（总损失 / 总样本数）
        avg_loss = total_loss / num_samples
        # 每10轮打印一次损失（观察收敛情况）
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}, 平均损失: {avg_loss:.4f}")

    # 4. 测试模型（用前10个样本验证）
    print("\n" + "=" * 50)
    print("模型测试（前10个样本）：")
    test_X = X[:, :10, :]  # 前10个样本的输入序列
    test_y = y[:10, :]  # 前10个样本的真实标签
    y_hat = rnn.forward(test_X)

    # 转换为标签（概率最大的类别）
    predictions = np.argmax(y_hat, axis=1)  # 预测标签
    true_labels = np.argmax(test_y, axis=1)  # 真实标签

    print(f"预测标签: {predictions}")
    print(f"真实标签: {true_labels}")
    accuracy = np.mean(predictions == true_labels)
    print(f"测试准确率: {accuracy:.2f}")

    # 5. 示例预测（自定义评论）
    print("\n" + "=" * 50)
    print("自定义评论预测：")

    def predict_sentiment(text):
        """预测单条评论的情感：正面/负面"""
        # 文本预处理（和训练时一致）
        text = str(text).strip()
        words = jieba.lcut(text)
        words = [word for word in words if word.strip()]

        # 构建序列向量（形状：(max_seq_len, 1, embedding_dim)，batch_size=1）
        seq = np.zeros((max_seq_len, 1, embedding_dim))
        for t in range(min(len(words), max_seq_len)):
            word = words[t]
            if word in word_to_idx:
                seq[t, 0, :] = word_vectors[word_to_idx[word]]

        # 模型预测
        y_hat = rnn.forward(seq)
        pred_label = np.argmax(y_hat, axis=1)[0]  # 取概率最大的类别
        confidence = y_hat[0, pred_label]  # 该类别的置信度
        return "正面" if pred_label == 1 else "负面", confidence

    # 测试2条示例评论
    test_examples = [
        "这家酒店环境很好，服务也很周到，下次还会再来。",
        "房间又小又脏，服务态度差，非常不满意。"
    ]
    for example in test_examples:
        sentiment, confidence = predict_sentiment(example)
        print(f"评论：{example}")
        print(f"预测：{sentiment}（置信度：{confidence:.4f}）\n")


# 运行训练
if __name__ == "__main__":
    train_rnn()