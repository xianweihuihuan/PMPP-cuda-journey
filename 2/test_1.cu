#include <cuda_runtime.h>

// 练习 1 (Exercise 1)
// 在本章中，我们实现了一个矩阵乘法核函数，该核函数让每个线程负责计算输出矩阵中的一个元素。在这个练习中，你将编写新的矩阵乘法核函数。请测试你编写的新核函数，并将其性能与本章示例中（每线程计算单元素）的核函数进行对比。
// a. 编写一个核函数，让每个线程负责计算输出矩阵的一整行 (one output matrix row)。请为这个设计写出对应的执行配置（Execution Configuration，即 <<<grid, block>>> 参数的设定）。
// b. 编写一个核函数，让每个线程负责计算输出矩阵的一整列 (one output matrix column)。请为这个设计写出对应的执行配置参数。
// c. 分析并论述这两种核函数设计各自的优缺点（Pros and Cons）。

__global__ void
matrixmulRow(float* M, int M_h, int M_w, float* N, int N_h, int N_w, float* P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < M_h) {
    for (int k = 0; k < N_w; ++k) {
      int val = 0.0f;
      for (int j = 0; j < M_w;++j){
        val += M[i * M_w + j] * N[j * N_w + k];
      }
      P[i * N_w + k] = val;
    }
  }
}

__global__ void
matrixmulRow(float* M, int M_h, int M_w, float* N, int N_h, int N_w, float* P){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N_w){
    for (int k = 0; k < M_h;++k){
      float val = 0.0f;
      for (int j = 0; j < N_h; ++j) {
        val += M[k * M_w + j] * N[j * N_w + i];
      }
      P[k * N_w + i] = val;
    }
  }
}