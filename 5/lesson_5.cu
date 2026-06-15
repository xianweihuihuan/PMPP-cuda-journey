#include <cuda_runtime.h>
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
    P[Row * width + ColStart + k * width] = vals[k];
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
      NDs[c][ty][tx] =
          N[(ty + i * TILE_WIDTH) * width + ColStart + c * TILE_WIDTH + tx];
    }
    for (int r = 0; r < COARSE_FACTOR; ++r) {
      int Row = RowStart + r * TILE_WIDTH;
      MDs[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];
      __syncthreads();
      for (int c = 0; c < COARSE_FACTOR; ++c) {
        int Col = ColStart + c * TILE_WIDTH;
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
