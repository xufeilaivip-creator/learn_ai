import numpy as np
import jieba  # 用于中文分词（先pip install jieba）


# 激活函数及其导数（不变）
def tanh(x):
    """双曲正切激活函数"""
    return np.tanh(x)


def tanh_derivative(h):
    """h 是 tanh(x) 的输出"""
    return 1 - h ** 2

# 需先确保已定义sigmoid激活函数（LSTM门控专用）
def sigmoid(x):
    """sigmoid激活函数：输出0-1，用于LSTM门控（控制信息流通）"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(h):
    """sigmoid导数：基于输出h计算（h = sigmoid(x)），用于反向传播"""
    return h * (1 - h)

def softmax(x):
    """softmax激活函数，用于输出层（避免指数溢出，减去每行最大值）"""
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# ------------- 3. 手动实现LSTM类（基于原RNN修改，核心新增门控和细胞状态）-------------
class ManualLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        """
        LSTM参数初始化（扩展RNN：新增4组门控权重+细胞状态相关参数）
        核心差异：LSTM有4个门控（遗忘/输入/候选细胞/输出），每组门控需输入→隐藏+隐藏→隐藏权重
        """
        self.hidden_size = hidden_size  # 隐藏层/细胞状态维度（二者维度一致）
        self.output_size = output_size

        # -------------------------- 1. 新增：LSTM门控权重（共8组：4个门控×2类权重）--------------------------
        # 遗忘门（f）：控制丢弃多少历史细胞状态
        self.W_xf = np.random.rand(input_size, hidden_size) * 0.01  # 输入→遗忘门权重
        self.W_hf = np.random.rand(hidden_size, hidden_size) * 0.01 # 隐藏→遗忘门权重
        self.b_f = np.zeros((1, hidden_size))  # 遗忘门偏置

        # 输入门（i）：控制保留多少新信息到细胞状态
        self.W_xi = np.random.rand(input_size, hidden_size) * 0.01  # 输入→输入门权重
        self.W_hi = np.random.rand(hidden_size, hidden_size) * 0.01 # 隐藏→输入门权重
        self.b_i = np.zeros((1, hidden_size))  # 输入门偏置

        # 候选细胞状态（c_tilde）：生成新的候选信息
        self.W_xc = np.random.rand(input_size, hidden_size) * 0.01  # 输入→候选细胞权重
        self.W_hc = np.random.rand(hidden_size, hidden_size) * 0.01 # 隐藏→候选细胞权重
        self.b_c = np.zeros((1, hidden_size))  # 候选细胞偏置

        # 输出门（o）：控制输出多少细胞状态到隐藏状态
        self.W_xo = np.random.rand(input_size, hidden_size) * 0.01  # 输入→输出门权重
        self.W_ho = np.random.rand(hidden_size, hidden_size) * 0.01 # 隐藏→输出门权重
        self.b_o = np.zeros((1, hidden_size))  # 输出门偏置

        # -------------------------- 2. 保留原RNN的输出层权重（隐藏→输出）--------------------------
        self.Why = np.random.rand(hidden_size, output_size) * 0.01
        self.by = np.zeros((1, output_size))

        # -------------------------- 3. 打印权重形状（调试用，确认维度）--------------------------
        print("=== LSTM权重形状 ===")
        print(f"遗忘门: W_xf({self.W_xf.shape}) | W_hf({self.W_hf.shape})")
        print(f"输入门: W_xi({self.W_xi.shape}) | W_hi({self.W_hi.shape})")
        print(f"候选细胞: W_xc({self.W_xc.shape}) | W_hc({self.W_hc.shape})")
        print(f"输出门: W_xo({self.W_xo.shape}) | W_ho({self.W_ho.shape})")
        print(f"输出层: Why({self.Why.shape})")

        # -------------------------- 4. 缓存字典（新增门控/细胞状态的中间结果，供反向传播用）--------------------------
        self.cache = {}

    def forward(self, x):
        """
        LSTM前向传播（核心新增：门控计算+细胞状态更新）
        输入x：同RNN → (seq_len, batch_size, input_size)
        输出y_hat：同RNN → (batch_size, output_size)
        """
        # 1. 解析输入形状（同RNN）
        seq_len, batch_size, _ = x.shape

        # 2. 新增：初始化隐藏状态h和细胞状态c（均为全0，LSTM需同时维护两个状态）
        h_prev = np.zeros((batch_size, self.hidden_size))  # 上一时间步隐藏状态
        c_prev = np.zeros((batch_size, self.hidden_size))  # 上一时间步细胞状态

        # 3. 缓存：保存每个时间步的中间结果（供反向传播用）
        self.cache['x'] = x  # 输入序列
        self.cache['h_states'] = [h_prev]  # 所有时间步的隐藏状态
        self.cache['c_states'] = [c_prev]  # 所有时间步的细胞状态
        self.cache['f_gates'] = []  # 所有时间步的遗忘门输出
        self.cache['i_gates'] = []  # 所有时间步的输入门输出
        self.cache['c_tildes'] = [] # 所有时间步的候选细胞输出
        self.cache['o_gates'] = []  # 所有时间步的输出门输出

        # 4. 逐时间步计算（核心修改：替换RNN的h更新为LSTM的门控+细胞状态更新）
        for t in range(seq_len):
            x_t = x[t]  # 当前时间步输入（同RNN）

            # -------------------------- 步骤1：计算4个门控和候选细胞（LSTM核心）--------------------------
            # 遗忘门（f_t）：sigmoid输出0-1，0=完全丢弃，1=完全保留
            f_t = sigmoid(np.dot(x_t, self.W_xf) + np.dot(h_prev, self.W_hf) + self.b_f)
            # 输入门（i_t）：sigmoid输出0-1，0=完全不保留新信息，1=完全保留
            i_t = sigmoid(np.dot(x_t, self.W_xi) + np.dot(h_prev, self.W_hi) + self.b_i)
            # 候选细胞（c_tilde_t）：tanh输出-1~1，生成新的候选信息
            c_tilde_t = tanh(np.dot(x_t, self.W_xc) + np.dot(h_prev, self.W_hc) + self.b_c)
            # 输出门（o_t）：sigmoid输出0-1，控制细胞状态对隐藏状态的贡献
            o_t = sigmoid(np.dot(x_t, self.W_xo) + np.dot(h_prev, self.W_ho) + self.b_o)

            # -------------------------- 步骤2：更新细胞状态（c_t）：遗忘旧信息+保留新信息--------------------------
            c_t = f_t * c_prev + i_t * c_tilde_t  # 核心公式：细胞状态的长期记忆更新

            # -------------------------- 步骤3：更新隐藏状态（h_t）：输出门控制+细胞状态激活--------------------------
            h_t = o_t * tanh(c_t)  # 核心公式：隐藏状态的短期输出（用于当前预测或传递到下一时间步）

            # -------------------------- 步骤4：缓存当前时间步的中间结果--------------------------
            self.cache['f_gates'].append(f_t)
            self.cache['i_gates'].append(i_t)
            self.cache['c_tildes'].append(c_tilde_t)
            self.cache['o_gates'].append(o_t)
            self.cache['h_states'].append(h_t)
            self.cache['c_states'].append(c_t)

            # -------------------------- 步骤5：更新前一时间步状态（供下一轮循环用）--------------------------
            h_prev = h_t
            c_prev = c_t

        # 5. 计算输出层（同RNN：用最后一个时间步的隐藏状态h_prev预测）
        y = np.dot(h_prev, self.Why) + self.by
        y_hat = softmax(y)
        self.cache['y_hat'] = y_hat  # 缓存预测结果

        return y_hat

    def backward(self, y_true):
        """
        LSTM反向传播（扩展RNN：新增细胞状态梯度+4组门控权重梯度计算）
        核心逻辑：从输出层→隐藏层→细胞状态→门控，反向传递误差，累加各时间步梯度
        """
        # 1. 从缓存取出前向传播的中间结果
        x = self.cache['x']
        y_hat = self.cache['y_hat']
        h_states = self.cache['h_states']  # [h0, h1, ..., hT]（T=seq_len）
        c_states = self.cache['c_states']  # [c0, c1, ..., cT]
        f_gates = self.cache['f_gates']    # [f1, ..., fT]
        i_gates = self.cache['i_gates']    # [i1, ..., iT]
        c_tildes = self.cache['c_tildes'] # [c_tilde1, ..., c_tildeT]
        o_gates = self.cache['o_gates']    # [o1, ..., oT]

        # 2. 解析输入形状（同RNN）
        seq_len, batch_size, input_size = x.shape
        hidden_size = self.hidden_size

        # 3. 初始化所有参数的梯度（扩展RNN：新增8组门控参数的梯度）
        # 门控权重梯度
        dW_xf = np.zeros_like(self.W_xf)
        dW_hf = np.zeros_like(self.W_hf)
        db_f = np.zeros_like(self.b_f)
        dW_xi = np.zeros_like(self.W_xi)
        dW_hi = np.zeros_like(self.W_hi)
        db_i = np.zeros_like(self.b_i)
        dW_xc = np.zeros_like(self.W_xc)
        dW_hc = np.zeros_like(self.W_hc)
        db_c = np.zeros_like(self.b_c)
        dW_xo = np.zeros_like(self.W_xo)
        dW_ho = np.zeros_like(self.W_ho)
        db_o = np.zeros_like(self.b_o)
        # 输出层权重梯度（同RNN）
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        # 4. 初始化反向传播的误差（新增细胞状态误差dc_next）
        dh_next = np.dot(y_hat - y_true, self.Why.T)  # 输出层误差传递到隐藏层（同RNN）
        dc_next = np.zeros((batch_size, hidden_size))  # 细胞状态的初始误差（最后一个时间步无后续，初始为0）

        # 5. 反向遍历时间步（从最后一个时间步T→0，同RNN但需处理细胞状态误差）
        for t in reversed(range(seq_len)):
            # 取出当前时间步的中间结果（注意索引：h_states[t+1]是当前h_t，h_states[t]是h_{t-1}）
            h_t = h_states[t+1]
            h_prev = h_states[t]
            c_t = c_states[t+1]
            c_prev = c_states[t]
            f_t = f_gates[t]
            i_t = i_gates[t]
            c_tilde_t = c_tildes[t]
            o_t = o_gates[t]
            x_t = x[t]

            # -------------------------- 步骤1：计算当前时间步的细胞状态误差（dc_t）--------------------------
            # 细胞状态误差来自两部分：1. 下一时间步传递的dc_next；2. 下一时间步传递的dh_next对c_t的依赖
            dc_t = dc_next * f_t + dh_next * o_t * tanh_derivative(tanh(c_t))

            # -------------------------- 步骤2：计算输出门（o_t）的梯度--------------------------
            do_t = dh_next * tanh(c_t) * sigmoid_derivative(o_t)
            dW_xo += np.dot(x_t.T, do_t)  # 输入→输出门权重梯度
            dW_ho += np.dot(h_prev.T, do_t)# 隐藏→输出门权重梯度
            db_o += np.sum(do_t, axis=0, keepdims=True)  # 输出门偏置梯度（按样本求和）

            # -------------------------- 步骤3：计算候选细胞（c_tilde_t）的梯度--------------------------
            dc_tilde_t = dc_t * i_t * tanh_derivative(c_tilde_t)
            dW_xc += np.dot(x_t.T, dc_tilde_t)  # 输入→候选细胞权重梯度
            dW_hc += np.dot(h_prev.T, dc_tilde_t)# 隐藏→候选细胞权重梯度
            db_c += np.sum(dc_tilde_t, axis=0, keepdims=True)  # 候选细胞偏置梯度

            # -------------------------- 步骤4：计算输入门（i_t）的梯度--------------------------
            di_t = dc_t * c_tilde_t * sigmoid_derivative(i_t)
            dW_xi += np.dot(x_t.T, di_t)  # 输入→输入门权重梯度
            dW_hi += np.dot(h_prev.T, di_t)# 隐藏→输入门权重梯度
            db_i += np.sum(di_t, axis=0, keepdims=True)  # 输入门偏置梯度

            # -------------------------- 步骤5：计算遗忘门（f_t）的梯度--------------------------
            df_t = dc_t * c_prev * sigmoid_derivative(f_t)
            dW_xf += np.dot(x_t.T, df_t)  # 输入→遗忘门权重梯度
            dW_hf += np.dot(h_prev.T, df_t)# 隐藏→遗忘门权重梯度
            db_f += np.sum(df_t, axis=0, keepdims=True)  # 遗忘门偏置梯度

            # -------------------------- 步骤6：更新下一个（前一时间步）的误差--------------------------
            dh_next = np.dot(do_t, self.W_ho.T) + np.dot(dc_tilde_t, self.W_hc.T) + np.dot(di_t, self.W_hi.T) + np.dot(df_t, self.W_hf.T)
            dc_next = dc_t  # 细胞状态误差直接传递到前一时间步

        # -------------------------- 步骤7：计算输出层权重梯度（同RNN）--------------------------
        h_final = h_states[-1]  # 最后一个时间步的隐藏状态
        dWhy = np.dot(h_final.T, y_hat - y_true)
        dby = np.sum(y_hat - y_true, axis=0, keepdims=True)

        # -------------------------- 步骤8：梯度平均（同RNN：除以批量大小，避免批量影响）--------------------------
        batch_size = x.shape[1]
        # 门控参数梯度平均
        dW_xf /= batch_size
        dW_hf /= batch_size
        db_f /= batch_size
        dW_xi /= batch_size
        dW_hi /= batch_size
        db_i /= batch_size
        dW_xc /= batch_size
        dW_hc /= batch_size
        db_c /= batch_size
        dW_xo /= batch_size
        dW_ho /= batch_size
        db_o /= batch_size
        # 输出层参数梯度平均
        dWhy /= batch_size
        dby /= batch_size

        # -------------------------- 步骤9：保存所有梯度（供参数更新用）--------------------------
        self.grads = {
            # 遗忘门
            'dW_xf': dW_xf, 'dW_hf': dW_hf, 'db_f': db_f,
            # 输入门
            'dW_xi': dW_xi, 'dW_hi': dW_hi, 'db_i': db_i,
            # 候选细胞
            'dW_xc': dW_xc, 'dW_hc': dW_hc, 'db_c': db_c,
            # 输出门
            'dW_xo': dW_xo, 'dW_ho': dW_ho, 'db_o': db_o,
            # 输出层
            'dWhy': dWhy, 'dby': dby
        }

    def update_parameters(self, learning_rate):
        """
        LSTM参数更新（扩展RNN：新增4组门控参数的更新）
        核心公式：同RNN → 新参数 = 旧参数 - 学习率 × 梯度
        """
        # -------------------------- 1. 更新门控参数（8组权重+4个偏置）--------------------------
        # 遗忘门
        self.W_xf -= learning_rate * self.grads['dW_xf']
        self.W_hf -= learning_rate * self.grads['dW_hf']
        self.b_f -= learning_rate * self.grads['db_f']
        # 输入门
        self.W_xi -= learning_rate * self.grads['dW_xi']
        self.W_hi -= learning_rate * self.grads['dW_hi']
        self.b_i -= learning_rate * self.grads['db_i']
        # 候选细胞
        self.W_xc -= learning_rate * self.grads['dW_xc']
        self.W_hc -= learning_rate * self.grads['dW_hc']
        self.b_c -= learning_rate * self.grads['db_c']
        # 输出门
        self.W_xo -= learning_rate * self.grads['dW_xo']
        self.W_ho -= learning_rate * self.grads['dW_ho']
        self.b_o -= learning_rate * self.grads['db_o']

        # -------------------------- 2. 更新输出层参数（同RNN）--------------------------
        self.Why -= learning_rate * self.grads['dWhy']
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
    rnn = ManualLSTM(input_size=embedding_dim, hidden_size=hidden_size, output_size=output_size)

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