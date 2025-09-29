import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，避免乱码
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# ----------------------
# 1. 手动实现简化版PCA
# ----------------------
class SimplePCA:
    def __init__(self, n_components):
        self.n_components = n_components  # 要保留的维度
        self.components = None  # 主成分（投影矩阵）

    def fit(self, X):
        """
        X: 输入数据，形状为 (样本数, 特征数)，这里即 (词数, 词向量维度)
        步骤：
        1. 数据中心化（减去均值）
        2. 计算协方差矩阵
        3. 对协方差矩阵做特征值分解
        4. 选取前n_components个最大特征值对应的特征向量作为主成分
        """
        # 步骤1：数据中心化（非常重要，否则PCA效果会受影响）
        self.mean_ = np.mean(X, axis=0)  # 计算每个特征的均值
        X_centered = X - self.mean_  # 减去均值，让数据围绕原点分布

        # 步骤2：计算协方差矩阵（衡量特征之间的相关性）
        # 协方差矩阵形状：(特征数, 特征数)
        cov_matrix = np.cov(X_centered.T)  # 注意要转置，因为cov默认行是样本

        # 步骤3：对协方差矩阵做特征值分解
        # 特征值：每个主成分的方差大小（越大表示该方向包含的信息越多）
        # 特征向量：主成分的方向（投影轴）
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 步骤4：按特征值从大到小排序，选取前n_components个特征向量
        # 得到排序索引（从大到小）
        sorted_indices = np.argsort(eigenvalues)[::-1]
        # 选取前n_components个特征向量作为主成分
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]

    def transform(self, X):
        """将数据投影到主成分上，得到降维结果"""
        X_centered = X - self.mean_  # 用训练时的均值中心化新数据
        # 矩阵乘法：(样本数, 原维度) × (原维度, 新维度) → (样本数, 新维度)
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """先训练再降维"""
        self.fit(X)
        return self.transform(X)


# ----------------------
# 2. 生成词向量（复用之前的简单Word2Vec模型）
# ----------------------
# 准备英文训练数据（避免中文乱码问题）
sentences = [
    "I love eating apples",
    "I love eating bananas",
    "I love eating oranges",
    "Cats love playing balls",
    "Dogs love playing frisbee",
    "Cats are cute",
    "Dogs are loyal",
    "Apples are fruits",
    "Bananas are fruits",
    "Oranges are fruits"
]

# 构建词汇表
from collections import defaultdict
def build_vocab(sentences):
    word_count = defaultdict(int)
    for sent in sentences:
        for word in sent.split():
            word_count[word] += 1
    vocab = list(word_count.keys())
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}
    return vocab, word_to_idx, idx_to_word

vocab, word_to_idx, idx_to_word = build_vocab(sentences)
vocab_size = len(vocab)

# 简单Word2Vec模型（生成10维词向量）
class SimpleWord2Vec:
    def __init__(self, vocab_size, embedding_dim=10):
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01  # 词向量矩阵
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01
        self.embedding_dim = embedding_dim

    def train(self, sentences, word_to_idx, epochs=500, lr=0.01, window_size=2):
        for _ in range(epochs):
            for sent in sentences:
                words = sent.split()
                for i, center_word in enumerate(words):
                    # 获取上下文词
                    ctx_words = []
                    for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
                        if j != i:
                            ctx_words.append(words[j])
                    # 训练每个上下文对
                    x = np.zeros(vocab_size)
                    x[word_to_idx[center_word]] = 1  # 中心词的One-Hot向量
                    for ctx_word in ctx_words:
                        y_idx = word_to_idx[ctx_word]
                        # 前向传播
                        h = np.dot(x, self.W1)
                        logits = np.dot(h, self.W2)
                        probs = np.exp(logits) / np.sum(np.exp(logits))
                        # 反向传播
                        d_logits = probs.copy()
                        d_logits[y_idx] -= 1
                        d_W2 = np.outer(h, d_logits)
                        d_h = np.dot(d_logits, self.W2.T)
                        d_W1 = np.outer(x, d_h)
                        # 更新参数
                        self.W1 -= lr * d_W1
                        self.W2 -= lr * d_W2

    def get_embeddings(self):
        return self.W1  # W1即为词向量矩阵

# 训练模型，得到10维词向量
w2v = SimpleWord2Vec(vocab_size, embedding_dim=10)
w2v.train(sentences, word_to_idx, epochs=500)
word_embeddings = w2v.get_embeddings()  # 形状：(词汇数, 10)
print("原始词向量形状：", word_embeddings.shape)


# ----------------------
# 3. 用手动实现的PCA降维到2D
# ----------------------
# 初始化PCA，降维到2D
pca = SimplePCA(n_components=2)
# 对词向量进行降维
embeddings_2d = pca.fit_transform(word_embeddings)
print("PCA降维后形状：", embeddings_2d.shape)  # 应为 (词汇数, 2)

# 打印主成分（前2个特征向量），看一下PCA找到的“主要方向”
print("\nPCA主成分（前2个特征向量）：")
print(pca.components)


# ----------------------
# 4. 可视化降维结果
# ----------------------
plt.figure(figsize=(12, 8))
for i in range(len(embeddings_2d)):
    word = idx_to_word[i]
    x, y = embeddings_2d[i]
    plt.scatter(x, y, s=80)
    plt.annotate(word, (x+0.03, y+0.03), fontsize=12)
plt.title("手动实现PCA的词向量降维结果（2D）")
plt.xlabel("主成分1（方差最大的方向）")
plt.ylabel("主成分2（方差次大的方向）")
plt.grid(alpha=0.3)
plt.show()
