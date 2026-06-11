#include <cuda_runtime.h>
#include <iostream>



// 练习 2：
// 矩阵-向量乘法接收一个输入矩阵 B 和一个输入向量 C，并生成一个输出向量 A。输出向量 A 中的每一个元素，等于输入矩阵 B 的对应行与向量 C 的点积：
// A_i = \sum_j (B_{i,j} * C_j)

// 为简化问题，假设矩阵为方阵（尺寸为 N * N），且所有元素均为单精度浮点数。请编写该操作的核函数（Kernel）以及对应的主机端封装函数（Host stub function）。

// 主机端函数需要接收以下四个参数：
// 1. 指向输出向量的指针
// 2. 指向输入矩阵的指针
// 3. 指向输入向量的指针
// 4. 每个维度的元素数量（即 N）

//核心要求：使用一个线程来计算输出向量中的一个元素。

__global__ void matrixmulKernel(float* M,
                                int M_h,
                                int M_w,
                                float* N,
                                int N_h,
                                int N_w,
                                float* P) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M_h && col < N_w) {
    float val = 0.0f;
    for (int i = 0; i < M_w; ++i) {
      val += M[row * M_w + i] * N[i * N_w + col];
    }
    P[row * N_w + col] = val;
  }
}

void matrixmul(float* M,
               int M_h,
               int M_w,
               float* N,
               int N_h,
               int N_w,
               float* P) {
  int M_size = M_h * M_w * sizeof(float);
  int N_size = N_h * N_w * sizeof(float);
  int P_size = M_h * N_w * sizeof(float);
  float* M_d;
  float* N_d;
  float* P_d;
  cudaError_t result;
  result = cudaMalloc((void**)&M_d, M_size);
  if (result != cudaSuccess) {
    printf("cuda malloc of M failed,reason:%s\n", cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&N_d, N_size);
  if (result != cudaSuccess) {
    printf("cuda malloc of  failed,reason:%s\n", cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&P_d, P_size);
  if (result != cudaSuccess) {
    printf("cuda malloc of P failed,reason:%s\n", cudaGetErrorString(result));
  }
  dim3 griddim((N_w + 15) / 16, (M_h + 15) / 16, 1);
  dim3 blockdim(16, 16, 1);
  cudaMemcpy(M_d, M, M_size, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N, N_size, cudaMemcpyHostToDevice);
  matrixmulKernel<<<griddim, blockdim>>>(M_d, M_h, M_w, N_d, N_h, N_w, P_d);
  cudaMemcpy(P, P_d, P_size, cudaMemcpyDeviceToHost);
  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
}

__global__ void sgemKernel(float* Pin, int w,int h,float* c,int il,float* out,int ol){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < ol){
    float val = 0.0f;
    for (int j = 0; j < w; ++j) {
      val += Pin[i * w + j] * c[j];
    }
    out[i] = val;
  }
}

void sgem(float* Pin, int w, int h, float* c, int il, float* out, int ol) {
  int Pin_size = w * h * sizeof(float);
  int c_size = il * sizeof(float);
  int out_size = ol * sizeof(float);
  float* Pin_d;
  float* c_d;
  float* out_d;
  cudaError_t result;
  result = cudaMalloc((void**)&Pin_d, Pin_size);
  if (result != cudaSuccess) {
    printf("cuda malloc of Pin failed,reason:%s\n", cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&c_d, c_size);
  if (result != cudaSuccess) {
    printf("cuda malloc of c failed,reason:%s\n", cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&out_d, out_size);
  if (result != cudaSuccess) {
    printf("cuda malloc of out failed,reason:%s\n", cudaGetErrorString(result));
  }
  dim3 griddim((ol + 255) / 256, 1, 1);
  dim3 blockdim(256, 1, 1);
  cudaMemcpy(Pin_d, Pin, Pin_size, cudaMemcpyHostToDevice);
  cudaMemcpy(c_d, c, c_size, cudaMemcpyHostToDevice);
  sgemKernel<<<griddim, blockdim>>>(Pin_d, w, h, c_d, il, out_d, ol);
  cudaMemcpy(out, out_d, out_size, cudaMemcpyDeviceToHost);
  cudaFree(Pin_d);
  cudaFree(c_d);
  cudaFree(out_d);
}
