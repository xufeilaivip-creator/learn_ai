import time
import numpy as np


from mmul import matmul_py, matmul_np
from mmul_cpp_cu import mmul_cpp, mmul_cuda


def run_with(func, *args):
    start = time.time()
    res = func(*args)
    end = time.time()
    cost = (end - start) * 1000
    print(f"{func.__name__} {cost:.2f} ms")
    return res


seed = 42
low = 0.0
high = 1.0

m = 500
n = 256
p = 256

rng = np.random.default_rng(seed)
arr_a = rng.uniform(low=low, high=high, size=(m, n))
arr_b = rng.uniform(low=low, high=high, size=(n, p))
a = arr_a.flatten().tolist()
b = arr_b.flatten().tolist()
wpa = rng.uniform(low=low, high=high, size=(20, 30)).flatten().tolist()
wpb = rng.uniform(low=low, high=high, size=(30, 40)).flatten().tolist()
mmul_cuda.matmul_cuda(wpa, wpb, 20, 30, 40)


# Pure Python
c1 = run_with(matmul_py, a, b, m, n, p)
# Numpy
c2 = run_with(matmul_np, arr_a, arr_b)
# Pure C++
c3 = run_with(mmul_cpp.matmul_cpp, a, b, m, n, p)
# Pure Cuda
c4 = run_with(mmul_cuda.matmul_cuda, a, b, m, n, p)

# Check The Result
c2 = c2.flatten().tolist()
print(len(c1), len(c2), len(c3), len(c4))
assert np.allclose(c1, c2)
assert np.allclose(c1, c3)
assert np.allclose(c1, c4)
