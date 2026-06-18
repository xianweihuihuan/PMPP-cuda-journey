#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16
__global__ void matrixMulKernel(float* M, float* N, float* P, int width) {
  __shared__ float Msd[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nsd[TILE_WIDTH][TILE_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  float val = 0.0f;
  for (int i = 0; i < width / TILE_WIDTH; ++i) {
    if (Row < width && i * TILE_WIDTH + tx < width) {
      Msd[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];
    } else {
      Msd[ty][tx] = 0.0f;
    }
    if (i * TILE_WIDTH + ty < width && Col < width) {
      Nsd[ty][tx] = N[(i * TILE_WIDTH + ty) * width + Col];
    } else {
      Nsd[ty][tx] = 0;
    }
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) {
      val += Msd[ty][k] * Nsd[k][tx];
    }
    __syncthreads();
  }
  if (Row < width && Col < width) {
    P[Row * width + Col] = val;
  }
}


__global__ void matrixMulExKernel(float* M,float* N,float* p,int width,int size){
  extern __shared__ float Msd_Nsd[];
  float* Msd = (float*)Msd_Nsd;
  float* Nsd = (float*)Msd_Nsd + size;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  float val = 0.0f;
  for (int i = 0; i < width / blockDim.x; ++i){
    if (Row < width && i * blockDim.x + tx < width) {
      Msd[ty * blockDim.x + tx] = M[Row * width + i * blockDim.x + tx];
    } else {
      Msd[ty * blockDim.x + tx] = 0.0f;
    }
    if (i * blockDim.y + ty < width && Col < width) {
      Nsd[ty * blockDim.x + tx] = N[(i * blockDim.x + ty) * width + Col];
    } else {
      Nsd[ty * blockDim.x + tx] = 0;
    }
    __syncthreads();
    for (int k = 0; k < blockDim.x; ++k) {
      val += Msd[ty * blockDim.x + k] * Nsd[k * blockDim.x + tx];
    }
    __syncthreads();
  }
  if(Row < width && Col < width){
    p[Row * width + Col] = val;
  }
}

bool checkCuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    std::cout << message << " failed, reason: " << cudaGetErrorString(result)
              << "\n";
    return false;
  }
  return true;
}

bool matrixMul(float* M_h, float* N_h, float* P_h, int width) {
  float* M_d;
  float* N_d;
  float* P_d;
  int size = width * width * sizeof(float);

  if (!checkCuda(cudaMalloc((void**)&M_d, size), "cuda malloc of M_d")) {
    return false;
  }
  if (!checkCuda(cudaMalloc((void**)&N_d, size), "cuda malloc of N_d")) {
    cudaFree(M_d);
    return false;
  }
  if (!checkCuda(cudaMalloc((void**)&P_d, size), "cuda malloc of P_d")) {
    cudaFree(M_d);
    cudaFree(N_d);
    return false;
  }

  checkCuda(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice),
            "copy M_h to M_d");
  checkCuda(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice),
            "copy N_h to N_d");

  dim3 blockdim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 griddim((width + TILE_WIDTH - 1) / TILE_WIDTH,
               (width + TILE_WIDTH - 1) / TILE_WIDTH,
               1);

  matrixMulKernel<<<griddim, blockdim>>>(M_d, N_d, P_d, width);
  checkCuda(cudaGetLastError(), "matrixMulKernel launch");
  checkCuda(cudaDeviceSynchronize(), "matrixMulKernel execution");

  checkCuda(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost),
            "copy P_d to P_h");

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
  return true;
}

bool matrixMulEx(float* M_h, float* N_h, float* P_h, int width) {
  float* M_d;
  float* N_d;
  float* P_d;
  int size = width * width * sizeof(float);

  if (!checkCuda(cudaMalloc((void**)&M_d, size), "cuda malloc of M_d")) {
    return false;
  }
  if (!checkCuda(cudaMalloc((void**)&N_d, size), "cuda malloc of N_d")) {
    cudaFree(M_d);
    return false;
  }
  if (!checkCuda(cudaMalloc((void**)&P_d, size), "cuda malloc of P_d")) {
    cudaFree(M_d);
    cudaFree(N_d);
    return false;
  }

  checkCuda(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice),
            "copy M_h to M_d");
  checkCuda(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice),
            "copy N_h to N_d");

  dim3 blockdim(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 griddim((width + TILE_WIDTH - 1) / TILE_WIDTH,
               (width + TILE_WIDTH - 1) / TILE_WIDTH,
               1);
  int sharedElements = TILE_WIDTH * TILE_WIDTH;
  int sharedBytes = 2 * sharedElements * sizeof(float);

  matrixMulExKernel<<<griddim, blockdim, sharedBytes>>>(
      M_d, N_d, P_d, width, sharedElements);
  checkCuda(cudaGetLastError(), "matrixMulExKernel launch");
  checkCuda(cudaDeviceSynchronize(), "matrixMulExKernel execution");

  checkCuda(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost),
            "copy P_d to P_h");

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
  return true;
}

void matrixMulCpu(float* M, float* N, float* P, int width) {
  for (int row = 0; row < width; ++row) {
    for (int col = 0; col < width; ++col) {
      float val = 0.0f;
      for (int k = 0; k < width; ++k) {
        val += M[row * width + k] * N[k * width + col];
      }
      P[row * width + col] = val;
    }
  }
}

float maxError(float* A, float* B, int width) {
  float err = 0.0f;
  for (int i = 0; i < width * width; ++i) {
    float diff = A[i] - B[i];
    if (diff < 0.0f) {
      diff = -diff;
    }
    if (diff > err) {
      err = diff;
    }
  }
  return err;
}

void printTopLeft(const char* title, float* data, int width, int count) {
  std::cout << title << "\n";
  for (int row = 0; row < count; ++row) {
    for (int col = 0; col < count; ++col) {
      std::cout << data[row * width + col] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main() {
  const int width = TILE_WIDTH;
  const int size = width * width;
  float M[size];
  float N[size];
  float P_cpu[size];
  float P_static[size];
  float P_dynamic[size];

  for (int row = 0; row < width; ++row) {
    for (int col = 0; col < width; ++col) {
      M[row * width + col] = row + col + 1.0f;
      N[row * width + col] = row == col ? 2.0f : 1.0f;
      P_cpu[row * width + col] = 0.0f;
      P_static[row * width + col] = 0.0f;
      P_dynamic[row * width + col] = 0.0f;
    }
  }

  matrixMulCpu(M, N, P_cpu, width);
  matrixMul(M, N, P_static, width);
  matrixMulEx(M, N, P_dynamic, width);

  printTopLeft("CPU result top-left 4x4:", P_cpu, width, 4);
  printTopLeft("static shared memory result top-left 4x4:", P_static, width, 4);
  printTopLeft("dynamic shared memory result top-left 4x4:", P_dynamic, width, 4);

  std::cout << "static shared memory max error: "
            << maxError(P_cpu, P_static, width) << "\n";
  std::cout << "dynamic shared memory max error: "
            << maxError(P_cpu, P_dynamic, width) << "\n";

  cudaDeviceReset();
  return 0;
}
