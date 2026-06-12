#include <cuda_stdint.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

__global__ void AddKernal(float* A, float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
  float* A_d;
  float* B_d;
  float* C_d;
  int size = sizeof(float) * n;
  cudaError_t result;
  result = cudaMalloc((void**)&A_d, size);
  if (result != cudaSuccess) {
    printf("cuda malloc of A_d failed,reason : %s\n",
           cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&B_d, size);
  if (result != cudaSuccess) {
    printf("cuda malloc of B_d failed,reason : %s\n",
           cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&C_d, size);
  if (result != cudaSuccess) {
    printf("cuda malloc of C_d failed,reason : %s\n",
           cudaGetErrorString(result));
  }
  printf("the location of A_d is %p\n", A_d);
  printf("the location of B_d is %p\n", B_d);
  printf("the location of C_d is %p\n", C_d);
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  auto start = std::chrono::high_resolution_clock::now();
  AddKernal<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  double seconds = elapsed.count();
  double bytes = 3.0 * size;
  double gbps = bytes / seconds / 1e9;

  printf("cuda add time: %f ms\n", seconds * 1000.0);
  printf("cuda add rate: %f GB/s\n", gbps);
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

void vecAddSerial(float* A_h, float* B_h, float* C_h, int n) {
  for (int i = 0; i < n; i++) {
    C_h[i] = A_h[i] + B_h[i];
  }
}

int main() {
  float* A_h = (float*)malloc(100000000 * sizeof(float));
  float* B_h = (float*)malloc(100000000 * sizeof(float));
  float* C_h = (float*)malloc(100000000 * sizeof(float));
  int n = 100000000;
  int size = sizeof(float) * n;

  for (int i = 0; i < n; i++) {
    A_h[i] = (float)rand() / RAND_MAX;
    B_h[i] = (float)rand() / RAND_MAX;
  }

  memset(C_h, 0, size);

  auto start = std::chrono::high_resolution_clock::now();
  vecAddSerial(A_h, B_h, C_h, n);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  double seconds = elapsed.count();
  double bytes = 3.0 * size;
  double gbps = bytes / seconds / 1e9;

  printf("serial add time: %f ms\n", seconds * 1000.0);
  printf("serial add rate: %f GB/s\n", gbps);

  memset(C_h, 0, size);

  vecAdd(A_h, B_h, C_h, n);
  free(A_h);
  free(B_h);
  free(C_h);
}