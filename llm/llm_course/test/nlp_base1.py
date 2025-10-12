# 1. 导入库
import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import re

# 2. 加载数据（示例格式）
data = pd.read_csv(R'D:\UGit\learn_ai\llm\llm_course\test\ChnSentiCorp_htl_all.csv')  # 假设包含'review'和'sentiment'列
# 正向评价标1，负向标0
print(data.head(3))

# 处理缺失值
data['review'] = data['review'].fillna('')  # 将NaN值替换为空字符串
data1=data['review']
# 加载停用词列表
with open('stopwords-zh.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())


# 3. 数据预处理
# 3.1 分词
def cut_words(text):
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(jieba.cut(text))
data['review'] = data['review'].apply(cut_words)


def cut_words1(text):
    # 1. 分词并过滤停用词（得到词列表）
    words = jieba.cut(text)
    filtered_words = [word.strip() for word in words if word.strip() and word not in stopwords]  # 过滤空字符和停用词

    # 2. 按标点拆分句子（保留完整语义）
    # 用标点作为分隔点，记录每个句子的词索引
    sentence_indices = [0]  # 第一个句子从索引0开始
    for i, word in enumerate(filtered_words):
        # 如果当前词是标点，标记句子结束
        if re.match(r'[。！？；,.!?;"]', word):
            sentence_indices.append(i + 1)  # 下一个句子从i+1开始

    # 3. 根据索引拆分出句子（每个句子是词列表）
    sentences = []
    for i in range(len(sentence_indices) - 1):
        start = sentence_indices[i]
        end = sentence_indices[i + 1]
        sentence = filtered_words[start:end]
        if sentence:  # 跳过空句子
            sentences.append(sentence)

    # 处理最后一个句子（如果没有标点结尾）
    last_start = sentence_indices[-1]
    if last_start < len(filtered_words):
        last_sentence = filtered_words[last_start:]
        if last_sentence:
            sentences.append(last_sentence)

    return sentences  # 返回：[[词1, 词2...], [词1, 词2...]]（句子列表，每个句子是词列表）

# 处理所有评论，拆平嵌套结构
all_sentences = []
# 遍历apply返回的每个评论的句子列表
for sentences_in_review in data1.apply(cut_words1):
    all_sentences.extend(sentences_in_review)  # 把每个评论的句子列表拆平添加

# 处理所有评论，将每条评论拆分成多个单句，再汇总成一个大列表
#all_sentences.extend(data1.apply(cut_words1))

# 3.2 特征抽取 使用Tfid
#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(data['review'])
#y = data['label']

# 3.2 特征抽取
# 使用Word2Vec生成词向量
print("all_sentences[:3]: ",all_sentences[:3])
word2vec_model = Word2Vec(all_sentences, vector_size=100, window=5, min_count=1, workers=4)
print(f'len(sentences): ', len(all_sentences))
print(f'len(word2vec_model.wv): ', len(word2vec_model.wv))

# 将每条评论转换为词向量的平均值
def get_sentence_vector(sentence):
    words = [word for word in sentence if word in word2vec_model.wv]
    if not words:
        return np.zeros(word2vec_model.vector_size)
    word_vectors = np.array([word2vec_model.wv[word] for word in words])
    return word_vectors.mean(axis=0)

X = np.array([get_sentence_vector(sentence) for sentence in data['review']])
y = data['label']

#print(X[0])

# 4. 模型训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Word2Vec生成词向量标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 贝叶斯模型   accuracy:0.7303732303732303
#clf = MultinomialNB()
#clf.fit(X_train, y_train)
# 逻辑回归模型  accuracy:0.8693693693693694
clf = LogisticRegression()  # 创建逻辑回归模型实例
clf.fit(X_train, y_train)  # 训练逻辑回归模型

# 5. 模型评估
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy: ", accuracy)




print(data.head(3))
print(data.shape)


