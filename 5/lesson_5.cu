#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define TILE_WIDTH 16
#define COARSE_FACTOR 4

__global__ void matrixMulKernel(float* M, float* N, float* P, int width) {
  __shared__ float MDs[TILE_WIDTH][TILE_WIDTH];
  __shared__ float NDs[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int ColStart = bx * (TILE_WIDTH * COARSE_FACTOR) + tx;
  float vals[COARSE_FACTOR];
  for (int k = 0; k < COARSE_FACTOR; ++k) {
    vals[k] = 0.0f;
  }
  for (int i = 0; i < width / TILE_WIDTH; ++i) {
    MDs[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int Col = ColStart + c * TILE_WIDTH;
      NDs[ty][tx] = N[(i * TILE_WIDTH + ty) * width + Col];
      __syncthreads();
      for (int k = 0; k < TILE_WIDTH; ++k) {
        vals[c] += MDs[ty][k] * NDs[k][tx];
      }
      __syncthreads();
    }
  }
  for (int k = 0; k < COARSE_FACTOR; ++k) {
    P[Row * width + ColStart + k * TILE_WIDTH] = vals[k];
  }
}

__global__ void matrixMul2DKernel(float* M, float* N, float* P, int width) {
  __shared__ float MDs[TILE_WIDTH][TILE_WIDTH];
  __shared__ float NDs[COARSE_FACTOR][TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int RowStart = by * TILE_WIDTH * COARSE_FACTOR + ty;
  int ColStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;
  float vals[COARSE_FACTOR][COARSE_FACTOR];
  for (int r = 0; r < COARSE_FACTOR; ++r) {
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      vals[r][c] = 0.0f;
    }
  }
  for (int i = 0; i < width / TILE_WIDTH; ++i) {
    for (int c = 0; c < COARSE_FACTOR; c++) {
      // ColStart 已含 +tx，这里不能再加 tx（否则列号偏移翻倍）
      NDs[c][ty][tx] =
          N[(ty + i * TILE_WIDTH) * width + ColStart + c * TILE_WIDTH];
    }
    for (int r = 0; r < COARSE_FACTOR; ++r) {
      int Row = RowStart + r * TILE_WIDTH;
      MDs[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];
      __syncthreads();
      for (int c = 0; c < COARSE_FACTOR; ++c) {
        //int Col = ColStart + c * TILE_WIDTH;
        for (int k = 0; k < TILE_WIDTH; k++) {
          vals[r][c] += MDs[ty][k] * NDs[c][k][tx];
        }
      }
      __syncthreads();
    }
  }
  for (int r = 0; r < COARSE_FACTOR; ++r) {
    int Row = RowStart + r * TILE_WIDTH;
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int Col = ColStart + c * TILE_WIDTH;
      P[Row * width + Col] = vals[r][c];
    }
  }
}


bool checkCuda(cudaError_t result, const char* msg) {
  if (result != cudaSuccess) {
    std::cout << msg << " failed: " << cudaGetErrorString(result) << "\n";
    return false;
  }
  return true;
}

void matrixMulCpu(const float* M, const float* N, float* P, int width) {
  for (int row = 0; row < width; ++row) {
    for (int col = 0; col < width; ++col) {
      float v = 0.0f;
      for (int k = 0; k < width; ++k) {
        v += M[row * width + k] * N[k * width + col];
      }
      P[row * width + col] = v;
    }
  }
}

float maxError(const float* A, const float* B, int n) {
  float err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d = fabsf(A[i] - B[i]);
    if (d > err) err = d;
  }
  return err;
}

int main() {
  // width 必须是 TILE_WIDTH * COARSE_FACTOR = 64 的倍数（两个 kernel 都没有边界检查）
  const int width = 1024;
  const int n = width * width;
  const int bytes = n * sizeof(float);

  float* M_h = new float[n];
  float* N_h = new float[n];
  float* P_cpu = new float[n];
  float* P_1d = new float[n];
  float* P_2d = new float[n];

  for (int row = 0; row < width; ++row) {
    for (int col = 0; col < width; ++col) {
      M_h[row * width + col] = ((row * 3 + col) % 13) * 0.5f;
      N_h[row * width + col] = ((row + col * 7) % 11) * 0.25f;
    }
  }

  matrixMulCpu(M_h, N_h, P_cpu, width);

  float *M_d, *N_d, *P_d;
  checkCuda(cudaMalloc(&M_d, bytes), "malloc M_d");
  checkCuda(cudaMalloc(&N_d, bytes), "malloc N_d");
  checkCuda(cudaMalloc(&P_d, bytes), "malloc P_d");
  checkCuda(cudaMemcpy(M_d, M_h, bytes, cudaMemcpyHostToDevice), "copy M");
  checkCuda(cudaMemcpy(N_d, N_h, bytes, cudaMemcpyHostToDevice), "copy N");

  dim3 block(TILE_WIDTH, TILE_WIDTH);

  // 1D 粗化：x 方向每个 block 覆盖 64 列，y 方向覆盖 16 行
  dim3 grid1d(width / (TILE_WIDTH * COARSE_FACTOR), width / TILE_WIDTH);
  matrixMulKernel<<<grid1d, block>>>(M_d, N_d, P_d, width);
  checkCuda(cudaGetLastError(), "launch 1D");
  checkCuda(cudaDeviceSynchronize(), "sync 1D");
  checkCuda(cudaMemcpy(P_1d, P_d, bytes, cudaMemcpyDeviceToHost), "copy P 1D");

  // 2D 粗化：两个方向每个 block 都覆盖 64×64
  dim3 grid2d(width / (TILE_WIDTH * COARSE_FACTOR),
              width / (TILE_WIDTH * COARSE_FACTOR));
  matrixMul2DKernel<<<grid2d, block>>>(M_d, N_d, P_d, width);
  checkCuda(cudaGetLastError(), "launch 2D");
  checkCuda(cudaDeviceSynchronize(), "sync 2D");
  checkCuda(cudaMemcpy(P_2d, P_d, bytes, cudaMemcpyDeviceToHost), "copy P 2D");

  std::cout << "width = " << width << "\n";
  std::cout << "1D coarsening max error: " << maxError(P_cpu, P_1d, n) << "\n";
  std::cout << "2D coarsening max error: " << maxError(P_cpu, P_2d, n) << "\n";

  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
  delete[] M_h;
  delete[] N_h;
  delete[] P_cpu;
  delete[] P_1d;
  delete[] P_2d;
  cudaDeviceReset();
  return 0;
}
