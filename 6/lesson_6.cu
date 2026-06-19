#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

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

// ===================== host 端：验证与对拍 =====================

bool checkCuda(cudaError_t result, const char* msg) {
  if (result != cudaSuccess) {
    std::cout << msg << " failed: " << cudaGetErrorString(result) << "\n";
    return false;
  }
  return true;
}

float maxError(const float* A, const float* B, int n) {
  float err = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d = fabsf(A[i] - B[i]);
    if (d > err) err = d;
  }
  return err;
}

void conv1D_cpu(const float* M, const float* F, float* P, int r, int l) {
  for (int i = 0; i < l; ++i) {
    float v = 0.0f;
    for (int j = 0; j < 2 * r + 1; ++j) {
      int idx = i - r + j;
      if (idx >= 0 && idx < l) {
        v += M[idx] * F[j];
      }
    }
    P[i] = v;
  }
}

void conv2D_cpu(const float* M, const float* F, float* P, int r, int width) {
  int fw = 2 * r + 1;
  for (int row = 0; row < width; ++row) {
    for (int col = 0; col < width; ++col) {
      float v = 0.0f;
      for (int frow = 0; frow < fw; ++frow) {
        for (int fcol = 0; fcol < fw; ++fcol) {
          int inrow = row - r + frow;
          int incol = col - r + fcol;
          if (inrow >= 0 && inrow < width && incol >= 0 && incol < width) {
            v += M[inrow * width + incol] * F[frow * fw + fcol];
          }
        }
      }
      P[row * width + col] = v;
    }
  }
}

int main() {
  const int r = FILTER_RADIUS;   // 5x5 滤波器
  const int fw = 2 * r + 1;      // 5
  const int fsize = fw * fw;     // 25

  // 非对称滤波器，方便暴露下标/转置类错误
  float* F_h = new float[fsize];
  for (int i = 0; i < fw; ++i) {
    for (int j = 0; j < fw; ++j) {
      F_h[i * fw + j] = (i * fw + j + 1) * 0.01f;
    }
  }

  // ===================== 1D 卷积 =====================
  {
    const int l = 4194304;   // 4M，给 ncu 足够大的 1D 工作量
    float* M_h = new float[l];
    float* P_cpu = new float[l];
    float* P_gpu = new float[l];
    for (int i = 0; i < l; ++i) {
      M_h[i] = (i % 17) * 0.5f;
    }

    conv1D_cpu(M_h, F_h, P_cpu, r, l);

    float *M_d, *F_d, *P_d;
    cudaMalloc(&M_d, l * sizeof(float));
    cudaMalloc(&F_d, fw * sizeof(float));   // 取滤波器前 5 个元素当 1D filter
    cudaMalloc(&P_d, l * sizeof(float));
    cudaMemcpy(M_d, M_h, l * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, fw * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (l + block - 1) / block;
    convolution_1D_Kernel<<<grid, block>>>(M_d, F_d, P_d, r, l);
    checkCuda(cudaDeviceSynchronize(), "1D conv");
    cudaMemcpy(P_gpu, P_d, l * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "[1D] length=" << l
              << "  max error: " << maxError(P_cpu, P_gpu, l) << "\n";

    cudaFree(M_d);
    cudaFree(F_d);
    cudaFree(P_d);
    delete[] M_h;
    delete[] P_cpu;
    delete[] P_gpu;
  }

  // ===================== 2D 卷积 =====================
  {
    // width 取 OUT_TILE_DIM(=28) 的倍数：halo-tile kernel 无写回边界检查，需精确覆盖
    const int width = 1120;   // 28 * 40，给 ncu 足够大的 2D 工作量
    const int n = width * width;
    const int bytes = n * sizeof(float);

    float* M_h = new float[n];
    float* P_cpu = new float[n];
    float* P_gpu = new float[n];
    for (int row = 0; row < width; ++row) {
      for (int col = 0; col < width; ++col) {
        M_h[row * width + col] = ((row * 3 + col * 7) % 13) * 0.5f;
      }
    }

    conv2D_cpu(M_h, F_h, P_cpu, r, width);

    float *M_d, *F_d, *P_d;
    cudaMalloc(&M_d, bytes);
    cudaMalloc(&F_d, fsize * sizeof(float));
    cudaMalloc(&P_d, bytes);
    cudaMemcpy(M_d, M_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(F_d, F_h, fsize * sizeof(float), cudaMemcpyHostToDevice);

    // 同一个滤波器拷进三块常量内存
    cudaMemcpyToSymbol(F, F_h, fsize * sizeof(float));
    cudaMemcpyToSymbol(F_c1, F_h, fsize * sizeof(float));
    cudaMemcpyToSymbol(F_c2, F_h, fsize * sizeof(float));

    // (a) 基本 2D，滤波器在 global memory
    {
      dim3 block(16, 16);
      dim3 grid((width + 15) / 16, (width + 15) / 16);
      convolution_2D_Kernel<<<grid, block>>>(M_d, F_d, P_d, r, width);
      checkCuda(cudaDeviceSynchronize(), "2D global F");
      cudaMemcpy(P_gpu, P_d, bytes, cudaMemcpyDeviceToHost);
      std::cout << "[2D] global F     max error: " << maxError(P_cpu, P_gpu, n)
                << "\n";
    }
    // (b) 滤波器在 constant memory
    {
      dim3 block(16, 16);
      dim3 grid((width + 15) / 16, (width + 15) / 16);
      convolution_2D_Kernel_by_const<<<grid, block>>>(M_d, P_d, width);
      checkCuda(cudaDeviceSynchronize(), "2D const F");
      cudaMemcpy(P_gpu, P_d, bytes, cudaMemcpyDeviceToHost);
      std::cout << "[2D] const F      max error: " << maxError(P_cpu, P_gpu, n)
                << "\n";
    }
    // (c) tiled + halo cells（block=输入 tile 32，grid 按 OUT_TILE_DIM=28 划分）
    {
      dim3 block(IN_TILE_DIM, IN_TILE_DIM);
      dim3 grid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
      convolution_2D_kernel_by_const_shared1<<<grid, block>>>(M_d, P_d, width);
      checkCuda(cudaDeviceSynchronize(), "2D halo tile");
      cudaMemcpy(P_gpu, P_d, bytes, cudaMemcpyDeviceToHost);
      std::cout << "[2D] tiled halo   max error: " << maxError(P_cpu, P_gpu, n)
                << "\n";
    }
    // (d) tiled + cache halo（block=输出 tile 32，halo 走 global/cache）
    {
      dim3 block(TILE_DIM, TILE_DIM);
      dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
                (width + TILE_DIM - 1) / TILE_DIM);
      convolution_2D_kernel_by_const_shared<<<grid, block>>>(M_d, P_d, width);
      checkCuda(cudaDeviceSynchronize(), "2D cache halo");
      cudaMemcpy(P_gpu, P_d, bytes, cudaMemcpyDeviceToHost);
      std::cout << "[2D] cache halo   max error: " << maxError(P_cpu, P_gpu, n)
                << "\n";
    }

    cudaFree(M_d);
    cudaFree(F_d);
    cudaFree(P_d);
    delete[] M_h;
    delete[] P_cpu;
    delete[] P_gpu;
  }

  delete[] F_h;
  cudaDeviceReset();
  return 0;
}
