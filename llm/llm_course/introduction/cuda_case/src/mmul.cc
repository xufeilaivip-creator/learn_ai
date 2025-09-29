#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

namespace mmul {


template <typename T>
std::vector<T> matmul_cpp(const std::vector<T>& A, const std::vector<T>& B, int m, int n, int p){
  std::vector<T> out(m*p, 0.0f);
  for(int i = 0; i < m; i++)
    for(int j = 0; j < p; j++){
      T v = 0.0f;
      for(int k = 0; k < n; k++){
        v += A[i * n + k] * B[k * p + j];
      }
      out[i * p + j] = v;
    }
  return out;
}


} // mmul



PYBIND11_MODULE(mmul_cpp, m) {
  
  namespace py = pybind11;
  
  using namespace mmul;

  m.def("matmul_cpp", matmul_cpp<int>);
  m.def("matmul_cpp", matmul_cpp<float>);

}