# 1. 导入需要的工具
import nltk
import jieba
from nltk.util import ngrams  # nltk自带的n-gram生成工具
from collections import defaultdict  # 用来统计词频的字典
import random  # 用来处理概率预测

# 2. 下载nltk的基础数据（第一次运行需要，后续不用）
#nltk.download('punkt')  # 用于分割句子成单词（分词）

# 3. 准备训练数据（可以换成你自己的文本，比如小说、评论）
# 这里用一段简单的中文评论作为示例，实际用的时候可以加更多文本
train_text = """
这家店的火锅很好吃，辣度适中，食材很新鲜。
下次还会来这家店，推荐大家尝试他们的毛肚和鸭肠。
这家店的服务态度很好，服务员很热情，上菜速度也快。
不推荐这家店的甜品，味道一般，价格还贵。
这家店的环境不错，很干净，适合朋友聚餐。
"""

# 4. 文本预处理：分词（把句子拆成单个词，方便生成n-gram）
# 使用jieba进行中文分词
tokens = jieba.lcut(train_text)
# 打印一下分词结果，看看效果
print("分词后的词列表：")
print(tokens[:10])  # 只打印前10个词，避免输出太长
# 输出示例：['这', '家', '店', '的', '火锅', '很', '好吃', '，', '辣', '度']


# 5. 核心1：生成n-gram组合（这里以2-gram为例，n=2）
n = 2  # 可以改成3（3-gram）、4（4-gram），根据需求调整
generated_ngrams = list(ngrams(tokens, n))  # 生成所有相邻n个词的组合
# 打印前5个n-gram，看看结构
print(f"\n生成的前5个{n}-gram：")
print(generated_ngrams[:5])
# 输出示例（2-gram）：[('这', '家'), ('家', '店'), ('店', '的'), ('的', '火锅'), ('火锅', '很')]


# 6. 核心2：统计n-gram频率，构建"前n-1个词→下一个词"的概率映射
# 用defaultdict统计：key是"前n-1个词的元组"，value是"下一个词的列表"（出现多少次就加多少次，方便后续算概率）
ngram_freq = defaultdict(list)
for gram in generated_ngrams:
    # gram是（前n-1个词，下一个词）的结构，比如2-gram中gram=('这','家')，前1个词是('这',)，下一个词是'家'
    prefix = gram[:-1]  # 前n-1个词（前缀）
    next_word = gram[-1]  # 下一个词（后缀）
    ngram_freq[prefix].append(next_word)  # 把后缀添加到前缀对应的列表里

# 打印一个前缀的统计结果，比如前缀是('这',)（2-gram的前1个词）
print(f"\n前缀('这家',)对应的下一个词列表（统计结果）：")
print(ngram_freq.get(('这家',), []))  # 输出示例：['家', '家', '家', '家']（因为文本里"这家"出现了4次）


# 7. 核心3：用n-gram预测下一个词（根据前缀，选出现频率最高的下一个词）
def predict_next_word(prefix_tokens, ngram_freq, n):
    """
    输入前缀词列表（比如["这"]），返回预测的下一个词
    :param prefix_tokens: 前缀词列表，长度必须是n-1（比如2-gram需要前缀长度1）
    :param ngram_freq: 之前统计的n-gram频率字典
    :param n: n-gram的n值
    :return: 预测的下一个词
    """
    # 把前缀列表转成元组（因为字典的key是元组）
    prefix = tuple(prefix_tokens)
    # 检查前缀长度是否正确（必须是n-1）
    if len(prefix) != n-1:
        return f"错误：前缀长度必须是{n-1}（因为是{n}-gram）"
    # 从频率字典中获取该前缀对应的所有下一个词
    next_word_list = ngram_freq.get(prefix, [])
    if not next_word_list:  # 如果没有统计到该前缀，返回"未知"
        return "未知（没统计到该前缀）"
    # 统计下一个词的频率，选出现次数最多的（即众数）
    from collections import Counter
    word_count = Counter(next_word_list)
    predicted_word = word_count.most_common(1)[0][0]  # most_common(1)返回频率最高的(词,次数)，取第一个词
    return predicted_word


# 测试预测功能（2-gram为例）
test_prefix1 = ["味道"]  # 2-gram的前缀长度是1，输入["这"]
pred1 = predict_next_word(test_prefix1, ngram_freq, n=2)
print(f"\n输入前缀{test_prefix1}，预测的下一个词：{pred1}")  # 输出：家（因为"这"后面最常出现"家"）

test_prefix2 = ["这家", "店"]  # 如果改成3-gram（n=3），前缀长度是2，输入["这家","店"]
# 先重新生成3-gram的频率字典（把n改成3再跑一遍步骤5-6）
n_3 = 3
generated_3grams = list(ngrams(tokens, n_3))
ngram_freq_3 = defaultdict(list)
for gram in generated_3grams:
    prefix = gram[:-1]
    next_word = gram[-1]
    ngram_freq_3[prefix].append(next_word)
# 测试3-gram的预测
pred2 = predict_next_word(["这家", "店"], ngram_freq_3, n=3)
print(f"输入前缀['这家','店']（3-gram），预测的下一个词：{pred2}")  # 输出：的（因为文本里"这家店的"出现多次）