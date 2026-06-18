#include <cuda_runtime.h>

__global__ void convolution_1D_Kernel(float* M,
                                      float* F,
                                      float* P,
                                      int r,
                                      int l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < l) {
    float val = 0.0f;
    for (int j = 0; j < 2 * r + 1; ++j) {
      int idx = i - r + j;
      if (idx >= 0 && idx < l) {
        val += M[idx] * F[j];
      }
    }
    P[i] = val;
  }
}

__global__ void convolution_2D_Kernel(float* M,
                                      float* F,
                                      float* P,
                                      int r,
                                      int width) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  if (Row >= 0 && Row < width && Col >= 0 && Col < width) {
    float val = 0.0f;
    for (int frow = 0; frow < 2 * r + 1; ++frow) {
      for (int fcol = 0; fcol < 2 * r + 1; ++fcol) {
        int inrow = Row - r + frow;
        int incol = Col - r + fcol;
        if (inrow >= 0 && inrow < width && incol >= 0 && incol < width) {
          val += M[inrow * width + incol] * F[frow * (2 * r + 1) + fcol];
        }
      }
    }
    P[Row * width + Col] = val;
  }
}

#define FILTER_RADIUS 2
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__ void convolution_2D_Kernel_by_const(float* M, float* P, int width) {
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  if (Row >= 0 && Row < width && Col >= 0 && Col < width) {
    float val = 0.0f;
    for (int frow = 0; frow < 2 * FILTER_RADIUS + 1; ++frow) {
      for (int fcol = 0; fcol < 2 * FILTER_RADIUS + 1; ++fcol) {
        int inrow = Row - FILTER_RADIUS + frow;
        int incol = Col - FILTER_RADIUS + fcol;
        if (inrow >= 0 && inrow < width && incol >= 0 && incol < width) {
          val += M[inrow * width + incol] * F[frow][fcol];
        }
      }
    }
    P[Row * width + Col] = val;
  }
}

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))
__constant__ float F_c1[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__ void convolution_2D_kernel_by_const_shared1(float* N,
                                                       float* P,
                                                       int width) {
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
  int col = bx * OUT_TILE_DIM + tx - FILTER_RADIUS;
  int row = by * OUT_TILE_DIM + ty - FILTER_RADIUS;
  if (row >= 0 && row < width && col >= 0 && col < width) {
    N_s[ty][tx] = N[row * width + col];
  } else {
    N_s[ty][tx] = 0.0f;
  }
  __syncthreads();

  int tileCol = tx - FILTER_RADIUS;
  int tileRow = ty - FILTER_RADIUS;
  if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 &&
      tileRow < OUT_TILE_DIM) {
    float val = 0.0f;
    for (int frow = 0; frow < 2 * FILTER_RADIUS + 1; ++frow) {
      for (int fcol = 0; fcol < 2 * FILTER_RADIUS + 1; ++fcol) {
        int inrow = ty - FILTER_RADIUS + frow;
        int incol = tx - FILTER_RADIUS + fcol;
        val += N_s[inrow][incol] * F_c1[frow][fcol];
      }
    }
    P[row * width + col] = val;
  }
}

#define TILE_DIM 32
__constant__ float F_c2[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];
__global__ void convolution_2D_kernel_by_const_shared(float* N,
                                                      float* P,
                                                      int width) {
  int by = blockIdx.y;
  int bx = blockIdx.x;
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  __shared__ float N_s[TILE_DIM][TILE_DIM];
  int Col = bx * TILE_DIM + tx;
  int Row = by * TILE_DIM + ty;
  if (Row < width && Col < width) {
    N_s[ty][tx] = N[Row * width + Col];
  } else {
    N_s[ty][tx] = 0.0f;
  }
  __syncthreads();
  if (Col < width && Row < width) {
    float val = 0.0f;
    for (int frow = 0; frow < 2 * FILTER_RADIUS + 1; ++frow) {
      for (int fcol = 0; fcol < 2 * FILTER_RADIUS + 1; ++fcol) {
        if (ty - FILTER_RADIUS + frow >= 0 &&
            ty - FILTER_RADIUS + frow < TILE_DIM &&
            tx - FILTER_RADIUS + fcol >= 0 &&
            tx - FILTER_RADIUS + fcol < TILE_DIM) {
          val += N_s[ty - FILTER_RADIUS + frow][tx - FILTER_RADIUS + fcol] *
                 F_c2[frow][fcol];
        } else if (Row - FILTER_RADIUS + frow >= 0 &&
                   Row - FILTER_RADIUS + frow < width &&
                   Col - FILTER_RADIUS + fcol >= 0 &&
                   Col - FILTER_RADIUS + fcol < width) {
          val += N[(Row - FILTER_RADIUS + frow) * width + Col - FILTER_RADIUS +
                   fcol] *
                 F_c2[frow][fcol];
        }
      }
    }
    P[Row * width + Col] = val;
  }
}
