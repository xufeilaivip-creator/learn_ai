# 安装库：pip install hmmlearn
from hmmlearn import hmm
import numpy as np

# 场景：隐藏状态（天气）：0=晴天, 1=雨天
# 观测值（活动）：0=散步, 1=购物, 2=打扫

# 1. 准备训练数据（100天的观测序列）
np.random.seed(42)
observations = np.random.randint(0, 3, size=(100, 1))  # 每天一个活动（0-2）

# 2. 创建并训练HMM模型（使用CategoricalHMM更适合离散观测值）
# 新版本中CategoricalHMM对应原来的MultinomialHMM功能
model = hmm.CategoricalHMM(
    n_components=2,  # 2个隐藏状态（晴天/雨天）
    n_iter=100,      # 训练迭代次数
    random_state=42
)

# 训练模型（输入需要是二维数组）
model.fit(observations)

# 3. 查看模型参数
print("训练后初始状态概率（晴天/雨天的初始概率）：")
print(model.startprob_)
print("\n状态转移概率（行→当前状态，列→下一个状态）：")
print(model.transmat_)
print("\n观测概率（行→状态，列→活动的概率）：")
print(model.emissionprob_)

# 4. 预测隐藏状态
new_observations = np.array([[0], [1], [2]])  # 新观测：散步→购物→打扫
hidden_states = model.predict(new_observations)
state_names = ["晴天", "雨天"]
print("\n根据观测序列预测的天气：")
for obs, state in zip(new_observations, hidden_states):
    print(f"活动{obs[0]} → 预测天气：{state_names[state]}")

# 5. 生成新序列（不再报错）
generated_obs, generated_states = model.sample(n_samples=5)
print("\n随机生成的5天序列：")
for obs, state in zip(generated_obs, generated_states):
    print(f"天气：{state_names[state]} → 活动{obs[0]}")
