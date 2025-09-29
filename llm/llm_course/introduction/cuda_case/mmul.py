import numpy as np


def matmul_py(a, b, m: int, n: int, p: int):
    """
    a: m x n
    b: n x p
    out: m x p
    """
    out = [0.0] * (m * p)
    for i in range(m):
        for j in range(p):
            v = 0
            for k in range(n):
                v += a[i * n + k] * b[k * p + j]
            out[i * p + j] = v
    return out


def matmul_np(a, b):
    return a @ b


if __name__ == "__main__":

    m = 2
    n = 4
    p = 3
    a = range(1, 9)
    b = range(1, 13)

    c = matmul_py(a, b, m, n, p)
    for i in range(m):
        print([c[i*p + j] for j in range(p)])

    a = np.arange(1, 9, dtype=np.float32).reshape(m, n)
    b = np.arange(1, 13, dtype=np.float32).reshape(n, p)
    c = matmul_np(a, b)
    print(c)