import numpy as np
import jieba  # 用于中文分词（先pip install jieba）
import torch
import torch.optim as optim
import torch.nn as nn


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

    # 4. 生成词向量（修改：更新正负词列表，包含分词后的实际词）
    word_vectors = np.zeros((vocab_size, embedding_dim))
    # 【关键修改】：加入分词后的组合词（从之前的分词结果里提取）
    positive_words = ["阳光", "阳光明媚", "晴朗", "天气晴朗", "舒服", "惬意", "暖和", "好看", "好", "干爽", "清新",
                      "温暖", "适合", "出游", "散步", "爬山", "晒被子", "花开"]
    negative_words = ["下雨", "大雾", "大风", "暴雨", "寒冷", "冰雹", "阴天", "台风", "沙尘暴", "霜冻", "麻烦",
                      "看不清", "响不停", "积水", "感冒", "砸坏", "压抑", "无聊", "很差", "冻坏"]

    for word, idx in word_to_idx.items():
        if word in positive_words:
            word_vectors[idx] = np.ones(embedding_dim) * 0.6 + np.random.randn(embedding_dim) * 0.05  # 增大正向偏向（0.6→更明显）
        elif word in negative_words:
            word_vectors[idx] = np.ones(embedding_dim) * -0.6 + np.random.randn(embedding_dim) * 0.05  # 增大负向偏向
        else:
            word_vectors[idx] = np.random.randn(embedding_dim) * 0.05  # 中性词随机波动稍大，不影响

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


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False
        )
        # 最后加一个线性层：把隐藏状态映射到类别
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (seq_len, batch_size, input_size)
        # 前向传播，得到所有时间步的输出和最后一个隐藏状态
        output, hn = self.rnn(x)  # hn shape: (num_layers, batch_size, hidden_size)

        # 取最后一层的最后一个时间步的隐藏状态（用于分类）
        last_hidden = hn[-1, :, :]  # shape: (batch_size, hidden_size)

        # 映射到类别
        logits = self.fc(last_hidden)  # shape: (batch_size, num_classes)
        return logits




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
    rnnModel = SimpleRNN(input_size=embedding_dim, hidden_size=hidden_size, output_size=output_size)
    # 损失函数：交叉熵损失（适合分类任务）
    criterion = nn.CrossEntropyLoss()
    # 优化器：Adam（负责参数更新，需要传入模型参数和学习率）
    optimizer = optim.Adam(rnnModel.parameters(), lr=0.001)


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
            batch_X_tensor = torch.from_numpy(batch_X).float()
            batch_y_tensor = torch.from_numpy(batch_y).float()

            # 前向传播：预测类别概率
            y_hat = rnnModel.forward(batch_X_tensor)
            # 计算损失（损失越小，预测越准）
            loss = criterion(y_hat, batch_y_tensor)
            total_loss += loss * (end_idx - i)  # 累计损失（乘以批量大小，最后平均）

            optimizer.zero_grad()
            # 反向传播：计算梯度
            loss.backward()
            # 优化器根据反向传播得到的梯度，更新所有参数（新参数 = 旧参数 - 学习率×梯度）
            optimizer.step()

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
    test_X_tensor = torch.from_numpy(test_X).float()
    test_y_tensor = torch.from_numpy(test_y).float()
    y_hat = rnnModel.forward(test_X_tensor)  # 模型预测

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
