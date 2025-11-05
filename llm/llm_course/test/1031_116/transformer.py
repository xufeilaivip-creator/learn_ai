import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------- 1. 位置编码（给序列加位置信息） --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 位置编码公式：PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        #              PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 分母项
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，方便广播到batch
        self.register_buffer('pe', pe)  # 不参与训练的参数

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # 加上位置编码（只取序列长度内的编码）
        return x


# -------------------------- 2. 多头自注意力（核心组件） --------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # 输入特征维度
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_model // num_heads  # 每个头的维度（必须整除）

        # Q、K、V的线性变换矩阵（共享参数）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 多头结果拼接后的线性变换
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)  # (batch_size, seq_len, d_model)

        # 1. 线性变换 + 分割多头
        q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # (batch, heads, seq_len, d_k)
        k = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数（缩放点积）
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)  # 归一化得到注意力权重

        # 3. 加权合并V
        output = torch.matmul(attn, v)  # (batch, heads, seq_len, d_k)

        # 4. 拼接多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch, seq_len, d_model)

        # 5. 最终线性变换
        output = self.w_o(output)
        return output


# -------------------------- 3. 前馈网络（每个位置独立的非线性变换） --------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)  # 升维
        self.fc2 = nn.Linear(hidden_dim, d_model)  # 降维
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.fc1(x)  # (batch, seq_len, hidden_dim)
        x = self.relu(x)
        x = self.fc2(x)  # (batch, seq_len, d_model)
        return x


# -------------------------- 4. Transformer的Encoder层（堆叠用） --------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)  # 多头自注意力
        self.ffn = FeedForward(d_model)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 层归一化1
        self.norm2 = nn.LayerNorm(d_model)  # 层归一化2
        self.dropout = nn.Dropout(0.1)  # 防止过拟合

    def forward(self, x):
        # 残差连接 + 层归一化（自注意力部分）
        attn_output = self.attn(x)
        x = self.norm1(x + self.dropout(attn_output))  # x + 注意力输出（残差），再归一化

        # 残差连接 + 层归一化（前馈网络部分）
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))  # x + 前馈输出（残差），再归一化
        return x


# -------------------------- 5. 完整的简化版Transformer（仅包含Encoder部分） --------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入（把词索引转成d_model维向量）
        self.pos_encoder = PositionalEncoding(d_model)  # 位置编码
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads) for _ in range(num_layers)  # 堆叠num_layers层Encoder
        ])

    def forward(self, x):
        # x: (batch_size, seq_len) 输入的词索引序列
        x = self.embedding(x)  # (batch, seq_len, d_model) 词嵌入
        x = self.pos_encoder(x)  # 加上位置编码

        # 经过多层Encoder
        for layer in self.layers:
            x = layer(x)
        return x  # 输出：(batch, seq_len, d_model) 包含上下文信息的序列向量


# -------------------------- 测试代码：用随机数据跑通Transformer --------------------------
if __name__ == "__main__":
    # 超参数
    vocab_size = 1000  # 词表大小（假设有1000个词）
    d_model = 64  # 特征维度（简化版，原版是512）
    num_heads = 2  # 注意力头数（简化版，原版是8）
    num_layers = 2  # Encoder层数

    # 初始化模型
    transformer = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers
    )

    # 随机生成输入：3个样本，每个样本是长度为10的词索引序列（值在0~999之间）
    input_seq = torch.randint(0, vocab_size, (3, 10))  # (batch_size=3, seq_len=10)
    print("输入形状：", input_seq.shape)  # torch.Size([3, 10])

    # 前向传播
    output = transformer(input_seq)
    print("输出形状：", output.shape)  # torch.Size([3, 10, 64]) （batch, seq_len, d_model）