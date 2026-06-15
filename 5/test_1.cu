#include <cuda_runtime.h>

#define TILE_WIDTH 16
__global__ void matrixMulKernel(float* M,float* N,float* P,int width){
  __shared__ float MDs[TILE_WIDTH][TILE_WIDTH];
  __shared__ float NDs[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Col = bx * blockDim.x + tx;
  int Row = by * blockDim.y + ty;
  float val = 0.0f;
  for (int i = 0; i < width / TILE_WIDTH; ++i) {
    MDs[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];
    NDs[tx][ty] = N[(bx * TILE_WIDTH + ty) * width  + i * TILE_WIDTH + tx];
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH;++k){
      val += MDs[ty][k] * NDs[k][tx];
    }
    __syncthreads();
  }
  P[Row * width + Col] = val;
}