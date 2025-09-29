import numpy as np
import jieba  # 用于中文分词（先pip install jieba）


# 激活函数及其导数（不变）
def tanh(x):
    """双曲正切激活函数"""
    return np.tanh(x)


def tanh_derivative(h):
    """h 是 tanh(x) 的输出"""
    return 1 - h ** 2


def softmax(x):
    """softmax激活函数，用于输出层（避免指数溢出，减去每行最大值）"""
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# 手动实现的RNN类（完全不变）
class ManualRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化RNN参数
        input_size: 输入特征维度（每个词向量的维度）
        hidden_size: 隐藏层大小
        output_size: 输出类别数
        """
        # 权重初始化（小随机数，避免梯度爆炸）
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01  # 输入→隐藏
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏→隐藏
        self.Why = np.random.randn(hidden_size, output_size) * 0.01  # 隐藏→输出
        print(f"Wxh(输入→隐藏): {self.Wxh.shape}")
        print(f"Whh(隐藏→隐藏): {self.Whh.shape}")
        print(f"Why(隐藏→输出): {self.Why.shape}")

        # 偏置初始化（初始为0）
        self.bh = np.zeros((1, hidden_size))  # 隐藏层偏置
        self.by = np.zeros((1, output_size))  # 输出层偏置

        # 存储中间结果，用于反向传播
        self.cache = {}

    def forward(self, x):
        """
        前向传播
        x: 输入序列，形状为 (时间步长, 批量大小, 输入特征数)
        """
        seq_len, batch_size, _ = x.shape  # 时间步=句子最大词数，批量=一次训练的样本数
        h = np.zeros((batch_size, self.Whh.shape[0]))  # 初始隐藏状态（全0）
        hidden_states = []  # 保存每个时间步的隐藏状态

        # 遍历每个时间步（每个词）
        for t in range(seq_len):
            x_t = x[t]  # 当前时间步的输入（批量内所有样本的第t个词向量）
            h = tanh(np.dot(x_t, self.Wxh) + np.dot(h, self.Whh) + self.bh)  # 更新隐藏状态
            hidden_states.append(h)

        # 用最后一个时间步的隐藏状态算输出（分类概率）
        y = np.dot(h, self.Why) + self.by
        y_hat = softmax(y)

        # 保存中间结果供反向传播用
        self.cache['x'] = x
        self.cache['hidden_states'] = hidden_states
        self.cache['h_final'] = h
        self.cache['y'] = y
        self.cache['y_hat'] = y_hat

        return y_hat

    def backward(self, y_true):
        """反向传播计算梯度（不变）"""
        x = self.cache['x']
        hidden_states = self.cache['hidden_states']
        h_final = self.cache['h_final']
        y_hat = self.cache['y_hat']

        seq_len, batch_size, input_size = x.shape
        # 初始化梯度（和权重同形状）
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # 输出层梯度（交叉熵损失的导数）
        dy = y_hat - y_true  # 预测值与真实值的误差
        dWhy = np.dot(h_final.T, dy)  # 隐藏→输出权重的梯度
        dby = np.sum(dy, axis=0, keepdims=True)  # 输出层偏置的梯度

        # 隐藏层梯度（从最后一个时间步反向推）
        dh_next = np.dot(dy, self.Why.T)  # 初始隐藏层误差

        for t in reversed(range(seq_len)):
            h = hidden_states[t]
            dh = dh_next * tanh_derivative(h)  # 当前时间步的隐藏状态梯度

            # 隐藏→隐藏权重的梯度（前一个时间步的隐藏状态贡献）
            if t > 0:
                h_prev = hidden_states[t - 1]
            else:
                h_prev = np.zeros_like(h)  # 第一个时间步没有前状态，用0
            dWhh += np.dot(h_prev.T, dh)

            # 输入→隐藏权重的梯度（当前时间步输入的贡献）
            x_t = x[t]
            dWxh += np.dot(x_t.T, dh)

            # 隐藏层偏置的梯度
            dbh += np.sum(dh, axis=0, keepdims=True)

            # 更新下一个（前一个时间步）的隐藏层误差
            dh_next = np.dot(dh, self.Whh.T)

        # 平均梯度（除以批量大小，避免批量影响梯度尺度）
        dWxh /= batch_size
        dWhh /= batch_size
        dWhy /= batch_size
        dbh /= batch_size
        dby /= batch_size

        self.grads = {
            'dWxh': dWxh,
            'dWhh': dWhh,
            'dWhy': dWhy,
            'dbh': dbh,
            'dby': dby
        }

    def update_parameters(self, learning_rate):
        """梯度下降更新参数（不变）"""
        self.Wxh -= learning_rate * self.grads['dWxh']
        self.Whh -= learning_rate * self.grads['dWhh']
        self.Why -= learning_rate * self.grads['dWhy']
        self.bh -= learning_rate * self.grads['dbh']
        self.by -= learning_rate * self.grads['dby']


# 辅助函数（不变）
def compute_loss(y_hat, y_true):
    """计算交叉熵损失（衡量预测值和真实值的差距）"""
    batch_size = y_hat.shape[0]
    # 加1e-10避免log(0)出错，最后平均到每个样本
    loss = -np.sum(y_true * np.log(y_hat + 1e-10)) / batch_size
    return loss


# -------------------------- 重点：文字样本生成函数（替换原来的随机数字样本） --------------------------
def generate_text_data(seq_len=6, embedding_dim=5):
    """
    生成有语义的文本样本：天气句子分类
    - 正面天气句子（标签1）：描述好天气，比如“阳光明媚”“适合出游”
    - 负面天气句子（标签0）：描述坏天气，比如“下雨麻烦”“大雾看不清”
    seq_len: 句子最大词数（不够补0，超过截断）
    embedding_dim: 每个词的向量维度
    返回：X（输入序列，shape=(seq_len, 样本数, embedding_dim)）、y（标签，one-hot编码）、texts（原始句子，方便查看）
    """
    # 1. 定义有语义的文本样本（2类，每类10个句子，共20个样本）
    positive_texts = [  # 正面天气（标签1）
        "今天阳光明媚很舒服",
        "周末天气晴朗适合出游",
        "下午微风拂面很惬意",
        "早晨天气暖和不用穿外套",
        "傍晚夕阳好看适合散步",
        "明天天气好可以去爬山",
        "最近天气干爽很舒服",
        "中午阳光好适合晒被子",
        "雨后天气清新空气好",
        "春天天气温暖花开了"
    ]
    negative_texts = [  # 负面天气（标签0）
        "今天下雨出门很麻烦",
        "早上大雾开车看不清路",
        "晚上刮大风窗户响不停",
        "昨天暴雨路上积水很多",
        "冬天天气寒冷容易感冒",
        "下午下冰雹砸坏了花盆",
        "阴天没有太阳很压抑",
        "台风天不能出门很无聊",
        "沙尘暴天气空气很差",
        "霜冻天气蔬菜都冻坏了"
    ]

    # 合并所有句子，标记标签（1=正面，0=负面）
    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    num_samples = len(texts)  # 总样本数：20

    # 2. 中文分词（把句子拆成词，比如“今天阳光明媚”→["今天", "阳光", "明媚"]）
    tokenized_texts = []
    for text in texts:
        words = jieba.lcut(text)  # jieba分词，返回词列表
        tokenized_texts.append(words)
    print("\n【所有文本样本及分词结果】")
    for i, (text, words) in enumerate(zip(texts, tokenized_texts)):
        print(f"样本{i+1}: {text} → 分词: {words} → 标签: {'正面天气' if labels[i]==1 else '负面天气'}")

    # 3. 构建词汇表（给每个词分配唯一索引，比如“今天”→0，“阳光”→1）
    word_to_idx = {}
    idx = 0
    for words in tokenized_texts:
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    vocab_size = len(word_to_idx)  # 词汇表大小（所有句子里不重复的词）
    print(f"\n【词汇表】（共{vocab_size}个不重复词）：{word_to_idx}")

    # 4. 生成词向量（简单版：用“词在词汇表的索引+小随机数”，让同类词向量接近一点）
    word_vectors = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_to_idx.items():
        # 给正面相关词（阳光、晴朗、舒服等）加微小正向值，负面相关词（下雨、大雾、麻烦等）加微小负向值
        if word in ["阳光", "晴朗", "舒服", "惬意", "暖和", "好看", "好", "干爽", "清新", "温暖"]:
            word_vectors[idx] = np.ones(embedding_dim) * 0.5 + np.random.randn(embedding_dim) * 0.01
        elif word in ["下雨", "大雾", "大风", "暴雨", "寒冷", "冰雹", "阴天", "台风", "沙尘暴", "霜冻", "麻烦"]:
            word_vectors[idx] = np.ones(embedding_dim) * -0.5 + np.random.randn(embedding_dim) * 0.01
        else:
            word_vectors[idx] = np.random.randn(embedding_dim) * 0.01

    # 5. 把句子转成RNN需要的输入格式：(seq_len, num_samples, embedding_dim)
    X = np.zeros((seq_len, num_samples, embedding_dim))  # 初始化全0
    for sample_idx in range(num_samples):  # 遍历每个样本
        words = tokenized_texts[sample_idx]  # 当前样本的词列表
        for time_step in range(min(len(words), seq_len)):  # 遍历每个词（不超过最大时间步）
            word = words[time_step]
            word_idx = word_to_idx[word]  # 词对应的索引
            X[time_step, sample_idx, :] = word_vectors[word_idx]  # 填充词向量

    # 6. 标签转one-hot编码（模型需要的格式：正面→[0,1]，负面→[1,0]）
    y = np.zeros((num_samples, 2))
    for i, label in enumerate(labels):
        y[i, label] = 1

    return X, y, texts, labels


# 训练函数（只改样本生成部分，其他不变）
def train_rnn():
    # 超参数（根据文字样本调整，更易训练）
    seq_len = 6  # 句子最大词数（比如“今天阳光明媚很舒服”是5个词，设6足够）
    embedding_dim = 5  # 每个词的向量维度（小一点，训练快）
    hidden_size = 8  # 隐藏层大小（匹配样本复杂度）
    output_size = 2  # 输出类别数（正面/负面）
    learning_rate = 0.1  # 学习率（稍大一点，加速收敛）
    epochs = 100  # 训练轮次
    batch_size = 5  # 批量大小（样本少，批量小一点）

    # -------------------------- 用文字样本替换原来的随机数字样本 --------------------------
    X, y, texts, true_labels = generate_text_data(seq_len=seq_len, embedding_dim=embedding_dim)
    num_samples = len(texts)  # 总样本数：20
    print(f"\n【模型输入格式】X.shape: {X.shape} → (时间步长={seq_len}, 样本数={num_samples}, 词向量维度={embedding_dim})")
    print(f"【标签格式】y.shape: {y.shape} → (样本数={num_samples}, 类别数={output_size})")

    # 创建RNN模型
    rnn = ManualRNN(input_size=embedding_dim, hidden_size=hidden_size, output_size=output_size)

    # 训练循环
    print("\n" + "="*50)
    print("开始训练：学习区分正面天气和负面天气句子")
    for epoch in range(epochs):
        total_loss = 0

        # 批量训练（遍历所有样本，每次取batch_size个）
        for i in range(0, num_samples, batch_size):
            # 截取当前批量（避免最后一批不足batch_size）
            end_idx = min(i + batch_size, num_samples)
            batch_X = X[:, i:end_idx, :]  # 批量输入
            batch_y = y[i:end_idx, :]     # 批量标签

            # 前向传播：预测类别概率
            y_hat = rnn.forward(batch_X)
            # 计算损失（损失越小，预测越准）
            loss = compute_loss(y_hat, batch_y)
            total_loss += loss * (end_idx - i)  # 累计损失（乘以批量大小，最后平均）

            # 反向传播：计算梯度
            rnn.backward(batch_y)
            # 更新参数：用梯度调整权重，让下次预测更准
            rnn.update_parameters(learning_rate)

        # 每10轮打印一次平均损失（观察训练进度）
        avg_loss = total_loss / num_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | 平均损失: {avg_loss:.4f}（损失越小越准）")

    # -------------------------- 测试模型：用前5个样本直观验证 --------------------------
    print("\n" + "="*50)
    print("【模型测试】取前5个样本看预测结果")
    test_idx = 5  # 测试前5个样本
    test_X = X[:, :test_idx, :]  # 测试输入
    test_y = y[:test_idx, :]     # 测试标签
    y_hat = rnn.forward(test_X)  # 模型预测

    # 转换结果：概率→标签（取概率最大的类别）
    pred_labels = np.argmax(y_hat, axis=1)  # 预测标签（0=负面，1=正面）
    pred_probs = [y_hat[i, pred_labels[i]] for i in range(test_idx)]  # 预测置信度（0-1，越近1越确定）

    # 打印详细结果（对应原始句子，直观看到预测对不对）
    print(f"{'样本':<4} {'原始句子':<15} {'真实标签':<8} {'预测标签':<8} {'置信度':<6}")
    print("-"*50)
    for i in range(test_idx):
        true_label = "正面" if true_labels[i] == 1 else "负面"
        pred_label = "正面" if pred_labels[i] == 1 else "负面"
        print(f"{i+1:<4} {texts[i]:<15} {true_label:<8} {pred_label:<8} {pred_probs[i]:.4f}")

    # 计算测试准确率
    accuracy = np.mean(pred_labels == true_labels[:test_idx])
    print(f"\n测试准确率: {accuracy:.2f}（1.0表示全对，0.0表示全错）")


# 运行训练
if __name__ == "__main__":
    train_rnn()