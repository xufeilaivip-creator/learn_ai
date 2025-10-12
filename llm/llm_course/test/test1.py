import numpy as np

import numpy as np

# 1. Sigmoid函数（常用于二分类输出层）
def sigmoid(x):
    """Sigmoid激活函数：将输入映射到(0, 1)区间"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(h):
    """Sigmoid导数：基于输出h计算（h = sigmoid(x)）"""
    return h * (1 - h)  # 公式：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))


# 2. ReLU函数（常用于隐藏层，缓解梯度消失）
def relu(x):
    """ReLU激活函数：x>0时输出x，否则输出0（稀疏激活特性）"""
    return np.maximum(0, x)

def relu_derivative(h):
    """ReLU导数：基于输出h计算（h = relu(x)）"""
    return (h > 0).astype(float)  # 公式：x>0时导数为1，x≤0时为0


# 3. Tanh函数（常用于隐藏层，输出范围对称）
def tanh(x):
    """Tanh激活函数：将输入映射到(-1, 1)区间（零中心化输出）"""
    return np.tanh(x)

def tanh_derivative(h):
    """Tanh导数：基于输出h计算（h = tanh(x)）"""
    return 1 - h **2  # 公式：tanh'(x) = 1 - tanh²(x)


# 4. Leaky ReLU函数（ReLU变体，解决死亡ReLU问题）
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU：x>0时输出x，否则输出alpha*x（保留小梯度）"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(h, alpha=0.01):
    """Leaky ReLU导数：基于输出h计算（h = leaky_relu(x)）"""
    return np.where(h > 0, 1, alpha)  # 公式：x>0时导数为1，x≤0时为alpha


h1,h2,h3=-0.5,0,0.5
derivative1=tanh_derivative(h1)
derivative2=tanh_derivative(h2)
derivative3=tanh_derivative(h3)
print(f"{h1}的derivative:{derivative1}")
print(f"{h2}的derivative:{derivative2}")
print(f"{h3}的derivative:{derivative3}")