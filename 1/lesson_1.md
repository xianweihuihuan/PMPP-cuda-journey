# PMPP 第一章学习笔记

## 1. 本章主线

PMPP 第一章的重点是建立 GPU 并行计算的基本认识：CPU 和 GPU 是异构处理器，CPU 更适合复杂控制逻辑和串行任务，GPU 更适合大量数据元素上的重复计算。

一个 CUDA 程序通常分成两部分：

- Host 端：运行在 CPU 上，负责申请内存、初始化数据、发起 kernel、回收结果。
- Device 端：运行在 GPU 上，负责执行大量并行线程。

本节代码使用向量加法作为最小例子：

```cpp
C[i] = A[i] + B[i]
```

这个计算天然是数据并行的，因为每个元素 `i` 的计算互不依赖。

## 2. 异构计算模型

CUDA 程序不是把整个 C/C++ 程序都放到 GPU 上运行，而是由 CPU 控制流程，在需要大量并行计算的位置启动 GPU kernel。

基本流程：

```text
CPU malloc host memory
CPU initialize input data
CPU cudaMalloc device memory
CPU cudaMemcpy host -> device
CPU launch CUDA kernel
GPU runs many threads
CPU cudaMemcpy device -> host
CPU cudaFree device memory
CPU free host memory
```

这个流程对应 `lesson_1.cu` 中的 `vecAdd` 函数。

## 3. CUDA 内存 API

### cudaMalloc

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size)
```

参数含义：

- `devPtr`：输出型参数，返回设备内存地址。
- `size`：需要申请的字节数。
- `return`：本次操作的错误码，成功时为 `cudaSuccess`。

作用：在 GPU device memory 上申请 `size` 字节空间。

当前代码示例：

```cpp
float* A_d;
int size = sizeof(float) * n;
cudaError_t result = cudaMalloc((void**)&A_d, size);
```

这里 `(void**)&A_d` 是因为 `cudaMalloc` 需要通过二级指针把申请到的 device 地址写回给调用者。

### cudaFree

```cpp
cudaError_t cudaFree(void* devPtr)
```

参数含义：

- `devPtr`：之前由 `cudaMalloc` 申请得到的设备指针。
- `return`：释放操作的错误码。

作用：释放 GPU device memory。

当前代码中申请了三个 device 数组，所以结束时需要分别释放：

```cpp
cudaFree(A_d);
cudaFree(B_d);
cudaFree(C_d);
```

### cudaMemcpy

```cpp
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```

参数含义：

- `dst`：目标地址。
- `src`：源地址。
- `count`：拷贝字节数。
- `kind`：拷贝方向。

常见方向：

```cpp
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaMemcpyDeviceToDevice
cudaMemcpyHostToHost
```

向量加法中需要两次输入拷贝和一次输出拷贝：

```cpp
cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
```

## 4. Kernel 函数

CUDA kernel 使用 `__global__` 修饰，表示这个函数从 host 端调用，在 device 端执行。

CUDA 常见函数修饰符：

| 修饰符 | 执行位置 | 调用位置 | 说明 |
| --- | --- | --- | --- |
| `__global__` | Device | Host | kernel 函数，使用 `<<<grid, block>>>` 启动，返回值必须是 `void`。 |
| `__device__` | Device | Device | 只能在 GPU 代码中被调用，常用于拆分 kernel 内部逻辑。 |
| `__host__` | Host | Host | 普通 CPU 函数。通常可以省略，因为默认就是 host 函数。 |

`__host__` 和 `__device__` 可以组合使用，表示同一个函数同时生成 CPU 版本和 GPU 版本：

```cpp
__host__ __device__
float add(float a, float b) {
  return a + b;
}
```

```cpp
__global__
void AddKernal(float* A, float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}
```

每个线程只负责一个元素的加法。

线程全局下标：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

含义：

- `threadIdx.x`：线程在当前 block 内的编号。
- `blockIdx.x`：当前 block 在 grid 中的编号。
- `blockDim.x`：每个 block 中的线程数量。

边界判断很重要：

```cpp
if (i < n)
```

因为启动的线程数通常会向上取整，可能比 `n` 多。如果不判断，最后一个 block 中多出来的线程会访问越界。

## 5. Kernel 启动配置

当前代码：

```cpp
AddKernal<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);
```

含义：

- `256`：每个 block 启动 256 个线程。
- `ceil(n / 256.0)`：需要的 block 数。

总线程数约等于：

```text
block_count * threads_per_block
```

只要总线程数不少于 `n`，并且 kernel 内部有 `if (i < n)`，就能覆盖全部元素且不越界。
