# PMPP 第三章学习笔记

## 1. 本章主线

PMPP 第三章的核心是把一维 CUDA 线程模型扩展到多维数据。很多实际问题不是简单的一维数组，而是图像、矩阵、张量这类二维或多维数据。

本章重点：

- 使用二维 grid 和二维 block 处理二维数据。
- 根据 `blockIdx`、`blockDim`、`threadIdx` 计算当前线程负责的数据位置。
- 在线程内部做边界判断，避免越界访问。
- 用图像处理和矩阵乘法理解数据并行。

当前代码 `lesson_2.cu` 中对应三个例子：

- RGB 图像转灰度图。
- 图像 blur 模糊。
- 方阵矩阵乘法。

## 2. 多维 Grid 和 Block

CUDA 的 kernel 启动配置可以是一维、二维或三维：

```cpp
dim3 griddim((width + 15) / 16, (height + 15) / 16, 1);
dim3 blockdim(16, 16, 1);

kernel<<<griddim, blockdim>>>(...);
```

这里每个 block 是 `16 x 16` 个线程，适合处理二维图像或矩阵。

二维线程坐标计算：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

含义：

- `col` 表示当前线程负责的列。
- `row` 表示当前线程负责的行。
- `x` 维通常对应列方向。
- `y` 维通常对应行方向。

## 3. 边界判断

因为 grid 大小通常通过向上取整得到，所以启动的线程数可能比真实数据元素更多。

例如：

```cpp
dim3 griddim((width + 15) / 16, (height + 15) / 16, 1);
```

当 `width` 或 `height` 不是 16 的整数倍时，最后一个 block 会有一些线程落在图像边界之外。

所以 kernel 内部必须判断：

```cpp
if (col < width && row < height) {
  ...
}
```

没有这个判断就可能访问非法内存。

## 4. RGB 转灰度图

RGB 图像通常每个像素有三个通道：

```text
R G B
```

在一维内存中，如果图像尺寸是 `width x height`，第 `row` 行、第 `col` 列像素的灰度图下标是：

```cpp
int grayOffset = row * width + col;
```

RGB 图像的下标是：

```cpp
int rgbOffset = grayOffset * 3;
```

读取三个通道：

```cpp
unsigned char r = Pin[rgbOffset];
unsigned char g = Pin[rgbOffset + 1];
unsigned char b = Pin[rgbOffset + 2];
```

灰度值计算：

```cpp
Pout[grayOffset] = 0.21 * r + 0.71 * g + 0.07 * b;
```

这体现了数据并行：每个线程只处理一个像素，像素之间互不依赖。

## 5. 图像 Blur

Blur 的思想是：每个输出像素等于它周围邻域像素的平均值。

当前代码中：

```cpp
#define BLUR_SIZE 1
```

表示以当前像素为中心，取 `3 x 3` 邻域。

遍历邻域偏移：

```cpp
for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
  for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
    int curRow = row + blurRow;
    int curCol = col + blurCol;
    ...
  }
}
```

由于边缘像素的邻域可能超出图像范围，所以需要判断：

```cpp
if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
  pixVal += in[curRow * w + curCol];
  pixels++;
}
```

最后计算平均值：

```cpp
out[row * w + col] = (unsigned char)(pixVal / pixels);
```

这个例子说明了 stencil 类计算的基本模式：每个线程负责一个输出元素，但需要读取周围多个输入元素。

## 6. 矩阵乘法

矩阵乘法中，输出矩阵 `P` 的一个元素由 `M` 的一行和 `N` 的一列点积得到：

```text
P[row][col] = sum(M[row][k] * N[k][col])
```

CUDA kernel 中每个线程负责计算一个 `P[row][col]`：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

边界判断：

```cpp
if (row < width && col < width) {
  ...
}
```

核心循环：

```cpp
float Pvalue = 0;
for (int k = 0; k < width; ++k) {
  Pvalue += M[row * width + k] * N[k * width + col];
}
P[row * width + col] = Pvalue;
```

矩阵在内存中按 row-major 方式存储：

```cpp
index = row * width + col
```

## 7. Host 外层函数

当前代码延续第一章的写法：kernel 外面包一层 host 函数。

外层函数负责：

- 计算数据大小。
- `cudaMalloc` 申请 device memory。
- `cudaMemcpy` 把 host 数据拷到 device。
- 配置 `grid` 和 `block`。
- 启动 kernel。
- `cudaMemcpy` 把结果拷回 host。
- `cudaFree` 释放 device memory。

例如矩阵乘法：

```cpp
void matrixmul(float* M_h, float* N_h, float* P_h, int width) {
  ...
  matrixmulKernel<<<griddim, blockdim>>>(M_d, N_d, P_d, width);
  ...
}
```

这样 `main()` 只需要调用普通 C++ 函数，不需要直接处理 device 指针。

## 8. 数据布局

本章几个例子都依赖一维数组表示二维数据。

二维坐标转一维下标：

```cpp
index = row * width + col
```

RGB 图像因为每个像素有 3 个通道，所以需要再乘以通道数：

```cpp
rgbOffset = (row * width + col) * 3
```

理解数据布局非常重要。线程坐标算对了，但一维下标算错，结果仍然会错。

## 9. 本章容易出错的点

常见错误：

- `row` 写成 `blockIdx.y * blockIdx.y + threadIdx.y`，应该乘 `blockDim.y`。
- 忘记边界判断，导致越界访问。
- RGB 图像忘记乘通道数。
- blur 中直接使用 `blurRow` / `blurCol` 访问数组，而不是使用 `row + blurRow` / `col + blurCol`。
- 矩阵乘法只判断 `row < width`，忘记判断 `col < width`。
- `cudaMalloc` 的 size 忘记乘 `sizeof(type)`。

## 10. 编译和运行

本机 RTX 4060 对应 compute capability 8.9，可以显式指定架构：

```bash
nvcc -arch=sm_89 2/lesson_2.cu -o /tmp/lesson_2
/tmp/lesson_2
```

如果不指定架构，某些环境下 kernel 可能没有正常执行，表现为输出全是 0。

## 11. 本章小结

PMPP 第三章的重点不是新的 CUDA API，而是把线程组织和数据布局联系起来。

需要掌握：

- 二维 grid / block 的使用方式。
- 从线程编号计算二维数据坐标。
- 从二维坐标计算一维数组下标。
- 图像和矩阵这类多维数据的并行化方式。
- 每个线程负责一个输出元素的基本 CUDA 编程模式。

这一章是后面学习 memory coalescing、shared memory tiling 和性能优化的基础。
