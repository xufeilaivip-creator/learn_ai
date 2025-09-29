import numpy as np

# 1. 数据准备
sentence = ["我", "爱", "吃", "苹果"]
vocab = list(set(sentence))  # 词汇表：["我", "爱", "吃", "苹果"]
word_to_idx = {w: i for i, w in enumerate(vocab)}  # 词→索引：{"我":0, "爱":1, "吃":2, "苹果":3}
idx_to_word = {i: w for w, i in word_to_idx.items()}  # 索引→词

# 生成训练数据：(中心词, 上下文词)对
# 对句子["我", "爱", "吃", "苹果"]，上下文窗口设为1（左右各看1个词）
train_data = []
for i in range(len(sentence)):
    center_word = sentence[i]
    # 上下文词：当前词的前1个和后1个（避免越界）
    context_words = []
    if i > 0:
        context_words.append(sentence[i-1])
    if i < len(sentence)-1:
        context_words.append(sentence[i+1])
    # 每个(中心词, 上下文词)都是一组训练样本
    for ctx in context_words:
        train_data.append((center_word, ctx))

# 2. 初始化参数
vec_dim = 2  # 词向量维度
vocab_size = len(vocab)  # 词汇表大小：4

# 词向量矩阵（输入层→隐藏层的权重）：vocab_size × vec_dim
W1 = np.random.randn(vocab_size, vec_dim) * 0.01  # 随机初始化小值
# 输出层权重（隐藏层→输出层）：vec_dim × vocab_size
W2 = np.random.randn(vec_dim, vocab_size) * 0.01

# 3. 训练（简化版梯度下降）
learning_rate = 0.01
epochs = 10000  # 训练轮次

for epoch in range(epochs):
    total_loss = 0
    for center_word, ctx_word in train_data:
        # 3.1 准备输入输出（One-Hot编码）
        # 中心词的One-Hot向量（1×4）
        center_idx = word_to_idx[center_word]
        x = np.zeros(vocab_size)
        x[center_idx] = 1  # 比如"爱"的One-Hot是[0,1,0,0]
        
        # 上下文词的索引（用于计算损失）
        ctx_idx = word_to_idx[ctx_word]
        
        # 3.2 前向传播
        h = x @ W1  # 隐藏层输出（1×2）：x是One-Hot，本质是取W1中对应行（中心词的向量）
        logits = h @ W2  # 输出层未激活值（1×4）
        # Softmax激活：将输出转为概率（预测每个词是上下文的概率）
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # 3.3 计算损失（交叉熵损失）：让正确上下文词的概率尽可能大
        loss = -np.log(probs[ctx_idx])
        total_loss += loss
        
        # 3.4 反向传播（简化版梯度计算）
        # 输出层误差
        d_logits = probs.copy()
        d_logits[ctx_idx] -= 1  # 对正确标签的梯度：概率-1
        
        # 更新W2和W1
        d_W2 = np.outer(h, d_logits)  # 外积计算梯度
        d_h = d_logits @ W2.T
        d_W1 = np.outer(x, d_h)
        
        W2 -= learning_rate * d_W2
        W1 -= learning_rate * d_W1
    
    # 每1000轮打印一次损失（应该逐渐减小）
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 4. 输出训练后的词向量（W1就是我们要的词向量矩阵）
print("\n训练后的词向量：")
for i in range(vocab_size):
    print(f"{idx_to_word[i]}: {W1[i]}")
