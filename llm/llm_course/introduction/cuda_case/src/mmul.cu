#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


using namespace std;


namespace mmul {


#define ROW_TILE_WIDTH 32
#define COL_TILE_WIDTH 32


template<typename T>
__global__
void matmul_cuda(const T *A, const T *B, T *out, int m, int n, int p) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < p) {
        T v = 0.0f;
        for (int k = 0; k < n; k++) {
            v += A[row * n + k] * B[k * p + col];
        }
        out[row * p + col] = v;
    }
}


template<typename T>
std::vector<T> matmul_cuda_wapper(std::vector<T>& A, std::vector<T>& B, int m, int n, int p) {

    const size_t ELEM_SIZE = sizeof(T);
    std::vector<T> C(m*p, 0.0f);

    auto a_size = (m*n) * ELEM_SIZE;
    auto b_size = (n*p) * ELEM_SIZE;
    auto c_size = (m*p) * ELEM_SIZE;

    T *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, a_size);
    cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_c, c_size);

    cudaMemcpy(d_a, A.data(), a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B.data(), b_size, cudaMemcpyHostToDevice);
    
    // dim grid
    dim3 numBlocks((p - 1) / COL_TILE_WIDTH + 1, (m - 1) / ROW_TILE_WIDTH + 1, 1);
    // dim block
    dim3 threadsPerBlock(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);
    matmul_cuda<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, m, n, p);

    cudaMemcpy(C.data(), d_c, c_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return C;
}


}  // namespace mmul


PYBIND11_MODULE(mmul_cuda, m) {
  namespace py = pybind11;
  using namespace mmul;

  m.def("matmul_cuda", matmul_cuda_wapper<float>);
  m.def("matmul_cuda", matmul_cuda_wapper<int>);
}
