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

int main() {
  auto printImage = [](const char* title,
                       const unsigned char* data,
                       int w,
                       int h) {
    std::cout << title << "\n";
    for (int row = 0; row < h; ++row) {
      for (int col = 0; col < w; ++col) {
        std::cout << static_cast<int>(data[row * w + col]) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  };

  auto printMatrix = [](const char* title, const float* data, int width) {
    std::cout << title << "\n";
    for (int row = 0; row < width; ++row) {
      for (int col = 0; col < width; ++col) {
        std::cout << data[row * width + col] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  };

  const int imageWidth = 4;
  const int imageHeight = 3;
  unsigned char rgbImage[imageWidth * imageHeight * CHANNLES];
  unsigned char grayImage[imageWidth * imageHeight] = {0};

  for (int row = 0; row < imageHeight; ++row) {
    for (int col = 0; col < imageWidth; ++col) {
      int offset = (row * imageWidth + col) * CHANNLES;
      rgbImage[offset] = static_cast<unsigned char>(row * 60);
      rgbImage[offset + 1] = static_cast<unsigned char>(col * 60);
      rgbImage[offset + 2] = static_cast<unsigned char>(120);
    }
  }

  colorToGrayscaleConversion(grayImage, rgbImage, imageWidth, imageHeight);
  printImage("grayscale result:", grayImage, imageWidth, imageHeight);

  const int blurWidth = 5;
  const int blurHeight = 5;
  unsigned char blurInput[blurWidth * blurHeight] = {
      0, 0, 0, 0, 0,
      0, 50, 50, 50, 0,
      0, 50, 255, 50, 0,
      0, 50, 50, 50, 0,
      0, 0, 0, 0, 0};
  unsigned char blurOutput[blurWidth * blurHeight] = {0};

  blur(blurInput, blurOutput, blurWidth, blurHeight);
  printImage("blur input:", blurInput, blurWidth, blurHeight);
  printImage("blur result:", blurOutput, blurWidth, blurHeight);

  const int matrixWidth = 3;
  float M[matrixWidth * matrixWidth] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f};
  float N[matrixWidth * matrixWidth] = {
      9.0f, 8.0f, 7.0f,
      6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f};
  float P[matrixWidth * matrixWidth] = {0.0f};

  matrixmul(M, N, P, matrixWidth);
  printMatrix("matrix M:", M, matrixWidth);
  printMatrix("matrix N:", N, matrixWidth);
  printMatrix("matrixmul result:", P, matrixWidth);

  cudaDeviceReset();
  return 0;
}
