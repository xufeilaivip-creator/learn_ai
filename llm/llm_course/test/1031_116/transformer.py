import numpy as np
import jieba
import torch
import torch.optim as optim
import torch.nn as nn
import math


# -------------------------- 1. 位置编码（Transformer必需） --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # 加上位置编码
        return x


# -------------------------- 2. 多头自注意力 --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output)


# -------------------------- 3. Transformer Encoder层 --------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output = self.attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


# -------------------------- 4. 完整Transformer模型（用于分类） --------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, num_heads=2, num_layers=2, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层（替代手动词向量）
        self.pos_encoder = PositionalEncoding(d_model)  # 位置编码
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)  # 分类头

    def forward(self, x):
        # x: (batch_size, seq_len) 词索引序列
        x = self.embedding(x) * math.sqrt(self.d_model)  # 词嵌入（乘缩放因子）
        x = self.pos_encoder(x)  # 加位置编码

        for layer in self.layers:
            x = layer(x)  # 经过多层Encoder

        # 取序列第一个位置的输出做分类（也可以取平均，这里简化）
        x = x[:, 0, :]  # (batch_size, d_model)
        logits = self.classifier(x)  # (batch_size, num_classes)
        return nn.functional.softmax(logits, dim=1)  # 输出概率


# -------------------------- 5. 文本样本生成函数（仅修改返回格式，保留词索引） --------------------------
def generate_text_data(seq_len=6):
    # 1. 定义正负样本句子
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

    # 2. 中文分词
    tokenized_texts = [jieba.lcut(text) for text in texts]
    print("\n【所有文本样本及分词结果】")
    for i, (text, words) in enumerate(zip(texts, tokenized_texts)):
        print(f"样本{i+1}: {text} → 分词: {words} → 标签: {'正面天气' if labels[i]==1 else '负面天气'}")

    # 3. 构建词汇表
    word_to_idx = {}
    idx = 0
    for words in tokenized_texts:
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    vocab_size = len(word_to_idx)
    print(f"\n【词汇表】（共{vocab_size}个不重复词）：{word_to_idx}")

    # 4. 句子转词索引序列（替换原词向量，作为Transformer输入）
    X = np.zeros((num_samples, seq_len), dtype=int)  # (样本数, 序列长度)，batch_first=True
    for i, words in enumerate(tokenized_texts):
        for t, word in enumerate(words[:seq_len]):  # 截断或补0
            X[i, t] = word_to_idx[word]

    # 5. 标签保持不变（0/1）
    y = np.array(labels)  # 直接用整数标签（CrossEntropyLoss需要）

    return X, y, texts, labels, vocab_size


# -------------------------- 6. 训练函数（适配Transformer输入格式） --------------------------
def train_transformer():
    # 超参数（适配Transformer）
    seq_len = 6
    d_model = 16  # Transformer特征维度
    num_heads = 2  # 注意力头数
    num_layers = 2  # Encoder层数
    num_classes = 2
    learning_rate = 0.001
    epochs = 100
    batch_size = 5

    # 生成数据（返回词索引序列和词汇表大小）
    X, y, texts, true_labels, vocab_size = generate_text_data(seq_len=seq_len)
    num_samples = len(texts)
    print(f"\n【模型输入格式】X.shape: {X.shape} → (样本数={num_samples}, 序列长度={seq_len})")
    print(f"【标签格式】y.shape: {y.shape} → (样本数={num_samples})")

    # 初始化Transformer模型
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes
    )

    # 损失函数和优化器（标签是整数，用CrossEntropyLoss）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    print("\n" + "="*50)
    print("开始训练：Transformer区分正面天气和负面天气句子")
    for epoch in range(epochs):
        total_loss = 0
        model.train()

        # 批量训练
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_X = X[i:end_idx]  # (batch_size, seq_len)
            batch_y = y[i:end_idx]  # (batch_size,)

            # 转成张量
            batch_X_tensor = torch.from_numpy(batch_X).long()  # 词索引是整数
            batch_y_tensor = torch.from_numpy(batch_y).long()

            # 前向传播
            y_hat = model(batch_X_tensor)
            loss = criterion(y_hat, batch_y_tensor)
            total_loss += loss.item() * (end_idx - i)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印损失
        avg_loss = total_loss / num_samples
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | 平均损失: {avg_loss:.4f}")

    # 测试模型
    print("\n" + "="*50)
    print("【模型测试】取前5个样本看预测结果")
    test_idx = 5
    test_X = X[:test_idx]
    test_X_tensor = torch.from_numpy(test_X).long()
    y_hat = model(test_X_tensor)

    # 转换预测结果
    pred_labels = torch.argmax(y_hat, dim=1).numpy()
    pred_probs = [y_hat[i, pred_labels[i]].item() for i in range(test_idx)]

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