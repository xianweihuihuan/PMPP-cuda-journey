#include <cuda_runtime.h>
#include <iostream>

#define CHANNLES 3
__global__ void colorToGrayscaleConversionKernel(unsigned char* Pout,
                                                 unsigned char* Pin,
                                                 int width,
                                                 int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height) {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * 3;
    unsigned char r = Pin[rgbOffset];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];
    Pout[grayOffset] = 0.21 * r + 0.71 * g + 0.07 * b;
  }
}

#define BLUR_SIZE 1
__global__ void blurKernel(unsigned char* in,
                           unsigned char* out,
                           int w,
                           int h) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < w && row < h) {
    int pixVal = 0;
    int pixels = 0;
    for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
      for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
        int curRow = row + blurRow;
        int curCol = col + blurCol;
        if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
          pixVal += in[curRow * w + curCol];
          pixels++;
        }
      }
    }
    out[row * w + col] = (unsigned char)(pixVal / pixels);
  }
}

__global__ void matrixmulKernel(float* M, float* N, float* P, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < width && col < width) {
    float Pvalue = 0;
    for (int k = 0; k < width; ++k) {
      Pvalue += M[row * width + k] * N[k * width + col];
    }
    P[row * width + col] = Pvalue;
  }
}

void colorToGrayscaleConversion(unsigned char* Pout_h,
                                unsigned char* Pin_h,
                                int width,
                                int height) {
  unsigned char* Pout_d;
  unsigned char* Pin_d;
  int Pin_size = width * height * CHANNLES;
  int Pout_size = width * height;
  cudaError_t result;
  result = cudaMalloc((void**)&Pout_d, Pout_size);
  if (result != cudaSuccess) {
    printf("cuda malloc of Pout failed: reason: %s",
           cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&Pin_d, Pin_size);
  if (result != cudaSuccess) {
    printf("cuda malloc of Pin failed: reason: %s", cudaGetErrorString(result));
  }
  cudaMemcpy(Pin_d, Pin_h, Pin_size, cudaMemcpyHostToDevice);
  dim3 griddim((width + 15) / 16, (height + 15) / 16, 1);
  dim3 blockdim(16, 16, 1);

  colorToGrayscaleConversionKernel<<<griddim, blockdim>>>(Pout_d, Pin_d, width,
                                                          height);
  cudaMemcpy(Pout_h, Pout_d, Pout_size, cudaMemcpyDeviceToHost);
  cudaFree(Pin_d);
  cudaFree(Pout_d);
}

void blur(unsigned char* in_h, unsigned char* out_h, int w, int h) {
  unsigned char* out_d;
  unsigned char* in_d;
  int size = w * h;
  cudaError_t result;
  result = cudaMalloc((void**)&out_d, size);
  if (result != cudaSuccess) {
    printf("cuda malloc of out failed: reason: %s", cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&in_d, size);
  if (result != cudaSuccess) {
    printf("cuda malloc of in failed: reason: %s", cudaGetErrorString(result));
  }
  cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);
  dim3 griddim((w + 15) / 16, (h + 15) / 16, 1);
  dim3 blockdim(16, 16, 1);
  blurKernel<<<griddim, blockdim>>>(in_d, out_d, w, h);
  cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);
  cudaFree(in_d);
  cudaFree(out_d);
}

void matrixmul(float* M_h, float* N_h, float* P_h, int width) {
  float* M_d;
  float* N_d;
  float* P_d;
  int size = width * width * sizeof(float);
  cudaError_t result;
  result = cudaMalloc((void**)&M_d, size);
  if (result != cudaSuccess) {
    printf("cuda malloc of M failed: reason: %s", cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&N_d, size);
  if (result != cudaSuccess) {
    printf("cuda malloc of N failed: reason: %s", cudaGetErrorString(result));
  }
  result = cudaMalloc((void**)&P_d, size);
  if (result != cudaSuccess) {
    printf("cuda malloc of P failed: reason: %s", cudaGetErrorString(result));
  }
  cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice);
  dim3 griddim((width + 15) / 16, (width + 15) / 16, 1);
  dim3 blockdim(16, 16, 1);
  matrixmulKernel<<<griddim, blockdim>>>(M_d, N_d, P_d, width);
  cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost);
  cudaFree(M_d);
  cudaFree(N_d);
  cudaFree(P_d);
}

// 只打印左上角一小块，避免大尺寸下刷屏
auto printImageCorner = [](const char* title,
                           const unsigned char* data,
                           int w,
                           int count) {
  std::cout << title << " (top-left " << count << "x" << count << ")\n";
  for (int row = 0; row < count; ++row) {
    for (int col = 0; col < count; ++col) {
      std::cout << static_cast<int>(data[row * w + col]) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
};

auto printMatrixCorner = [](const char* title,
                            const float* data,
                            int width,
                            int count) {
  std::cout << title << " (top-left " << count << "x" << count << ")\n";
  for (int row = 0; row < count; ++row) {
    for (int col = 0; col < count; ++col) {
      std::cout << data[row * width + col] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
};

int main() {
  // ===================== RGB 转灰度 / blur：1920x1080 =====================
  const int imageWidth = 1920;
  const int imageHeight = 1080;
  const int pixels = imageWidth * imageHeight;

  unsigned char* rgbImage = new unsigned char[pixels * CHANNLES];
  unsigned char* grayImage = new unsigned char[pixels];
  unsigned char* blurOutput = new unsigned char[pixels];

  for (int row = 0; row < imageHeight; ++row) {
    for (int col = 0; col < imageWidth; ++col) {
      int offset = (row * imageWidth + col) * CHANNLES;
      rgbImage[offset] = static_cast<unsigned char>((row + col) % 256);
      rgbImage[offset + 1] = static_cast<unsigned char>((row * 2) % 256);
      rgbImage[offset + 2] = static_cast<unsigned char>((col * 3) % 256);
    }
  }

  colorToGrayscaleConversion(grayImage, rgbImage, imageWidth, imageHeight);
  printImageCorner("grayscale result:", grayImage, imageWidth, 4);

  // blur 直接复用灰度图作为单通道输入
  blur(grayImage, blurOutput, imageWidth, imageHeight);
  printImageCorner("blur result:", blurOutput, imageWidth, 4);

  // ===================== 矩阵乘法：1024x1024 =====================
  const int matrixWidth = 1024;
  const int matrixSize = matrixWidth * matrixWidth;

  float* M = new float[matrixSize];
  float* N = new float[matrixSize];
  float* P = new float[matrixSize];

  for (int row = 0; row < matrixWidth; ++row) {
    for (int col = 0; col < matrixWidth; ++col) {
      M[row * matrixWidth + col] = ((row * 3 + col) % 13) * 0.5f;
      N[row * matrixWidth + col] = ((row + col * 7) % 11) * 0.25f;
      P[row * matrixWidth + col] = 0.0f;
    }
  }

  matrixmul(M, N, P, matrixWidth);
  printMatrixCorner("matrixmul result:", P, matrixWidth, 4);

  delete[] rgbImage;
  delete[] grayImage;
  delete[] blurOutput;
  delete[] M;
  delete[] N;
  delete[] P;
  cudaDeviceReset();
  return 0;
}
