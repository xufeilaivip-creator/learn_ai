from mmul_cpp_cu import mmul_cuda, mmul_cpp
import numpy as np


m = 2
n = 4
p = 3
a = list(range(1, 9))
b = list(range(1, 13))

c = np.array(a).reshape(m,n) @ np.array(b).reshape(n,p)
c1 = mmul_cpp.matmul_cpp(a, b, m, n, p)
c2 = mmul_cuda.matmul_cuda(a, b, m, n, p)

print(c.flatten().tolist())
print(c1)
print(c2)