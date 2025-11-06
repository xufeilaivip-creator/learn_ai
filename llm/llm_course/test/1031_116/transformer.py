import numpy as np
import jieba
import torch
import torch.optim as optim
import torch.nn as nn
import math


# -------------------------- 1. 位置编码（Positional Encoding） --------------------------
class PositionalEncoding(nn.Module):
    """
    作用：给序列中的每个词添加"位置信息"
    为什么需要？
    Transformer没有RNN的循环结构，无法通过"顺序计算"感知词的位置（比如"我爱你"和"你爱我"语义不同）。
    位置编码通过数学公式给不同位置的词分配独特的向量，让模型知道"词在句子中的位置"。
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 初始化位置编码矩阵：(max_len, d_model)，max_len是最大序列长度，d_model是词向量维度
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置索引：[0,1,2,...,max_len-1]，形状为(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 位置编码公式中的分母项：10000^(2i/d_model)，用指数函数计算更高效
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度用正弦函数（sin），奇数维度用余弦函数（cos）
        # 这样设计的好处：位置越近的词，编码越相似；且能外推到比max_len更长的序列
        pe[:, 0::2] = torch.sin(position * div_term)  # 0::2表示从0开始，步长为2（偶数索引）
        pe[:, 1::2] = torch.cos(position * div_term)  # 1::2表示从1开始，步长为2（奇数索引）
        
        # 增加一个维度：(1, max_len, d_model)，方便和batch数据广播相加
        pe = pe.unsqueeze(0)
        
        # 把位置编码注册为"非训练参数"（不需要更新，固定使用）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: 输入的词向量序列，形状为(batch_size, seq_len, d_model)
        # 给每个词向量加上对应的位置编码（只取输入序列长度的编码，避免浪费）
        x = x + self.pe[:, :x.size(1), :]  # self.pe[:, :seq_len, :]
        return x  # 输出：(batch_size, seq_len, d_model)，包含词向量+位置信息


# -------------------------- 2. 多头自注意力（Multi-Head Attention） --------------------------
class MultiHeadAttention(nn.Module):
    """
    作用：让模型同时关注序列中"不同类型的关联"（比如在天气分类中，既关注"阳光"与"舒服"的关联，也关注"适合"与"出游"的关联）
    原理：把注意力拆分成多个"头"（head），每个头独立计算注意力，最后合并结果，相当于并行捕捉多种关联
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # 词向量总维度（必须能被头数整除）
        self.num_heads = num_heads  # 注意力头数（比如2个头，同时捕捉2种关联）
        self.d_k = d_model // num_heads  # 每个头的维度（总维度/头数）

        # 定义Q、K、V的线性变换矩阵（把词向量映射到Q、K、V空间）
        # Q（Query）：当前词的"查询"向量（想找什么信息）
        # K（Key）：其他词的"键"向量（提供什么信息）
        # V（Value）：其他词的"值"向量（具体信息内容）
        self.w_q = nn.Linear(d_model, d_model)  # (d_model → d_model)
        self.w_k = nn.Linear(d_model, d_model)  # (d_model → d_model)
        self.w_v = nn.Linear(d_model, d_model)  # (d_model → d_model)

        # 多头结果合并后的线性变换（把多个头的输出合并成总维度）
        self.w_o = nn.Linear(d_model, d_model)  # (d_model → d_model)

    def forward(self, x):
        # x: 输入序列，形状为(batch_size, seq_len, d_model)
        batch_size = x.size(0)  # 批量大小

        # 1. 线性变换：把词向量映射到Q、K、V
        # 输出形状：(batch_size, seq_len, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. 拆分多头：把总维度d_model拆成num_heads个d_k
        # 形状变化：(batch_size, seq_len, d_model) → (batch_size, num_heads, seq_len, d_k)
        # view：重塑维度；transpose：交换seq_len和num_heads的位置，方便后续计算
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力分数（缩放点积注意力）
        # 公式：scores = (Q × K^T) / sqrt(d_k)
        # Q×K^T：计算Q和K的相似度（值越大，关联越强），形状：(batch_size, num_heads, seq_len, seq_len)
        # 除以sqrt(d_k)：防止d_k太大时，分数过大导致softmax梯度消失
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. 注意力权重：用softmax归一化分数，得到每个词对其他词的"关注程度"（0~1，和为1）
        attn = nn.functional.softmax(scores, dim=-1)  # 形状：(batch_size, num_heads, seq_len, seq_len)

        # 5. 加权合并：用注意力权重乘以V，得到每个词的"上下文信息"（关注的词贡献更多）
        # 形状：(batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attn, v)

        # 6. 拼接多头：把多个头的结果合并回总维度d_model
        # 形状变化：(batch_size, num_heads, seq_len, d_k) → (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 7. 最终线性变换：整合多头信息
        output = self.w_o(output)  # 形状：(batch_size, seq_len, d_model)
        return output


# -------------------------- 3. Transformer Encoder层 --------------------------
class TransformerEncoderLayer(nn.Module):
    """
    作用：Transformer的基本组成单元，每个层包含"多头自注意力"和"前馈网络"，负责提取序列的上下文特征
    为什么这样设计？
    自注意力负责捕捉词与词的关联（比如"阳光"和"舒服"），前馈网络负责增强每个词的特征表达（让模型学会更复杂的模式）
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)  # 多头自注意力模块

        # 前馈网络：每个位置独立的非线性变换（和序列中其他词无关，只增强单个词的特征）
        # 结构：升维→激活→降维（先把特征维度放大4倍，用ReLU引入非线性，再缩回去）
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # 升维：d_model → 4*d_model
            nn.ReLU(),  # 非线性激活（让模型能学习复杂模式）
            nn.Linear(4 * d_model, d_model)   # 降维：4*d_model → d_model
        )

        # 层归一化（LayerNorm）：稳定训练过程（让每一层的输入分布更稳定）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout：防止过拟合（随机丢弃部分神经元，让模型更鲁棒）
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: 输入序列，形状为(batch_size, seq_len, d_model)

        # 1. 多头自注意力 + 残差连接 + 层归一化
        # 残差连接：x + attn_output（保留原始输入，防止梯度消失）
        attn_output = self.attn(x)  # 注意力输出：(batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout(attn_output))  # 归一化：让训练更稳定

        # 2. 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)  # 前馈输出：(batch_size, seq_len, d_model)
        x = self.norm2(x + self.dropout(ffn_output))  # 再次归一化

        return x  # 输出：(batch_size, seq_len, d_model)，包含更丰富的上下文特征


# -------------------------- 4. 完整Transformer模型（用于分类） --------------------------
class SimpleTransformer(nn.Module):
    """
    完整的Transformer分类模型（仅包含Encoder部分，适合分类任务）
    整体流程：词嵌入→加位置编码→多层Encoder提取特征→分类头输出类别概率
    """
    def __init__(self, vocab_size, d_model=16, num_heads=2, num_layers=2, num_classes=2):
        super().__init__()
        self.d_model = d_model  # 词向量维度（所有层的特征维度）

        # 1. 词嵌入层（Embedding）：把离散的词索引（比如"阳光"→3）转成连续的向量（d_model维）
        # 作用：让计算机能理解词的语义（相似的词向量更接近）
        self.embedding = nn.Embedding(vocab_size, d_model)  # (vocab_size → d_model)

        # 2. 位置编码层：给词向量添加位置信息（前面定义的PositionalEncoding）
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. 多层Encoder：堆叠多个TransformerEncoderLayer（层数越多，能捕捉的特征越复杂）
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])

        # 4. 分类头（Classifier）：把最终的序列特征映射到类别（正面/负面天气）
        self.classifier = nn.Linear(d_model, num_classes)  # (d_model → num_classes)

    def forward(self, x):
        # x: 输入的词索引序列，形状为(batch_size, seq_len)（比如[[3,5,2], [1,4,0]]，每个数字是词的索引）

        # 步骤1：词嵌入 + 缩放
        # 形状变化：(batch_size, seq_len) → (batch_size, seq_len, d_model)
        # 乘以sqrt(d_model)：让嵌入向量的方差更稳定（和位置编码的量级匹配）
        x = self.embedding(x) * math.sqrt(self.d_model)

        # 步骤2：添加位置编码（让模型知道词的位置）
        x = self.pos_encoder(x)  # 形状不变：(batch_size, seq_len, d_model)

        # 步骤3：经过多层Encoder（逐层提取上下文特征）
        for layer in self.layers:
            x = layer(x)  # 每一层输出仍为：(batch_size, seq_len, d_model)

        # 步骤4：取序列的第一个位置作为"句子特征"（也可以取所有位置的平均，这里简化）
        # 形状：(batch_size, seq_len, d_model) → (batch_size, d_model)
        x = x[:, 0, :]  # 取第0个时间步的特征（可理解为"句子的整体特征"）

        # 步骤5：分类头输出类别得分（logits）
        logits = self.classifier(x)  # 形状：(batch_size, num_classes)（比如[[0.8, -0.2], ...]）

        # 步骤6：用softmax转成概率（0~1，和为1）
        return nn.functional.softmax(logits, dim=1)  # 形状：(batch_size, num_classes)


# -------------------------- 5. 文本样本生成函数（数据部分，保持不变） --------------------------
def generate_text_data(seq_len=6):
    # 1. 定义正负样本句子（天气分类：正面/负面）
    positive_texts = [
        "今天阳光明媚很舒服", "周末天气晴朗适合出游", "下午微风拂面很惬意",
        "早晨天气暖和不用穿外套", "傍晚夕阳好看适合散步", "明天天气好可以去爬山",
        "最近天气干爽很舒服", "中午阳光好适合晒被子", "雨后天气清新空气好", "春天天气温暖花开了"
    ]
    negative_texts = [
        "今天下雨出门很麻烦", "早上大雾开车看不清路", "晚上刮大风窗户响不停",
        "昨天暴雨路上积水很多", "冬天天气寒冷容易感冒", "下午下冰雹砸坏了花盆",
        "阴天没有太阳很压抑", "台风天不能出门很无聊", "沙尘暴天气空气很差", "霜冻天气蔬菜都冻坏了"
    ]

    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)
    num_samples = len(texts)

    # 2. 中文分词（把句子拆成词，比如"今天阳光明媚"→["今天", "阳光", "明媚"]）
    tokenized_texts = [jieba.lcut(text) for text in texts]
    print("\n【所有文本样本及分词结果】")
    for i, (text, words) in enumerate(zip(texts, tokenized_texts)):
        print(f"样本{i+1}: {text} → 分词: {words} → 标签: {'正面天气' if labels[i]==1 else '负面天气'}")

    # 3. 构建词汇表（给每个词分配唯一索引，比如"今天"→0，"阳光"→1）
    word_to_idx = {}
    idx = 0
    for words in tokenized_texts:
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    vocab_size = len(word_to_idx)
    print(f"\n【词汇表】（共{vocab_size}个不重复词）：{word_to_idx}")

    # 4. 句子转词索引序列（作为Transformer输入）
    X = np.zeros((num_samples, seq_len), dtype=int)  # (样本数, 序列长度)
    for i, words in enumerate(tokenized_texts):
        for t, word in enumerate(words[:seq_len]):  # 截断或补0（保证长度一致）
            X[i, t] = word_to_idx[word]

    # 5. 标签（0=负面，1=正面）
    y = np.array(labels)
    return X, y, texts, labels, vocab_size


# -------------------------- 6. 训练函数（适配Transformer） --------------------------
def train_transformer():
    # 超参数
    seq_len = 6  # 句子最大长度（词数）
    d_model = 16  # 词向量维度
    num_heads = 2  # 注意力头数
    num_layers = 2  # Encoder层数
    num_classes = 2  # 类别数（正面/负面）
    learning_rate = 0.001  # 学习率
    epochs = 100  # 训练轮次
    batch_size = 5  # 批量大小

    # 生成数据
    X, y, texts, true_labels, vocab_size = generate_text_data(seq_len=seq_len)
    num_samples = len(texts)
    print(f"\n【模型输入格式】X.shape: {X.shape} → (样本数={num_samples}, 序列长度={seq_len})")
    print(f"【标签格式】y.shape: {y.shape} → (样本数={num_samples})")

    # 初始化模型
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适合分类）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

    # 训练循环
    print("\n" + "="*50)
    print("开始训练：Transformer区分正面天气和负面天气句子")
    for epoch in range(epochs):
        total_loss = 0
        model.train()  # 训练模式

        # 批量训练
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_X = X[i:end_idx]  # 批量输入：(batch_size, seq_len)
            batch_y = y[i:end_idx]  # 批量标签：(batch_size,)

            # 转成PyTorch张量（词索引是整数，用long类型）
            batch_X_tensor = torch.from_numpy(batch_X).long()
            batch_y_tensor = torch.from_numpy(batch_y).long()

            # 前向传播：模型预测概率
            y_hat = model(batch_X_tensor)  # 形状：(batch_size, 2)

            # 计算损失（预测与真实标签的差距）
            loss = criterion(y_hat, batch_y_tensor)
            total_loss += loss.item() * (end_idx - i)  # 累计损失

            # 反向传播+参数更新
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

        # 每10轮打印平均损失
        avg_loss = total_loss / num_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | 平均损失: {avg_loss:.4f}")

    # 测试模型
    print("\n" + "="*50)
    print("【模型测试】取前5个样本看预测结果")
    test_idx = 5
    test_X = X[:test_idx]
    test_X_tensor = torch.from_numpy(test_X).long()
    y_hat = model(test_X_tensor)  # 预测概率

    # 转换预测结果（概率→标签）
    pred_labels = torch.argmax(y_hat, dim=1).numpy()  # 取概率最大的类别
    pred_probs = [y_hat[i, pred_labels[i]].item() for i in range(test_idx)]  # 预测置信度

    # 打印详情
    print(f"{'样本':<4} {'原始句子':<15} {'真实标签':<8} {'预测标签':<8} {'置信度':<6}")
    print("-"*50)
    for i in range(test_idx):
        true_label = "正面" if true_labels[i] == 1 else "负面"
        pred_label = "正面" if pred_labels[i] == 1 else "负面"
        print(f"{i+1:<4} {texts[i]:<15} {true_label:<8} {pred_label:<8} {pred_probs[i]:.4f}")

    # 计算准确率
    accuracy = np.mean(pred_labels == true_labels[:test_idx])
    print(f"\n测试准确率: {accuracy:.2f}")


if __name__ == "__main__":
    train_transformer()