# PMPP 第四章学习笔记

## 1. 本章主线

PMPP 第四章的重点是理解 GPU 的计算架构和线程调度方式。前面章节已经会写 kernel，本章开始解释 kernel 为什么能并行执行，以及硬件资源如何限制一个 kernel 的实际并行度。

本章核心问题：

- GPU 由多个 SM 组成。
- thread block 被分配到 SM 上执行。
- block 内线程会被划分成 warp。
- warp 是 SM 上实际调度执行的基本单位。
- SM 上的 shared memory、register、最大线程数、最大 block 数会限制 occupancy。

当前 `lesson_3.cu` 做的事情就是查询这些硬件属性：

```cpp
cudaGetDeviceCount(&deviceCount);
cudaGetDeviceProperties(&devprop, i);
```

## 2. GPU 和 SM

GPU 不是一个单独的大核心，而是由多个 Streaming Multiprocessor，也就是 SM 组成。

可以把 GPU 理解成：

```text
GPU = many SMs
SM = many CUDA cores + warp schedulers + registers + shared memory
```

一个 kernel 启动后，CUDA runtime 会把 grid 中的 thread blocks 分配到各个 SM 上执行。

重要规则：

- 一个 block 只会在一个 SM 上执行。
- 一个 block 不会被拆到多个 SM。
- 一个 block 开始执行后，通常会一直留在同一个 SM 上直到结束。
- 不同 block 之间没有固定执行顺序。
- 不同 block 之间不能在同一个 kernel 内做全局同步。

这也是为什么 CUDA 程序要求 block 之间尽量独立。

## 3. Thread Block 到 SM 的映射

CUDA 程序中我们写的是：

```cpp
kernel<<<gridDim, blockDim>>>(...);
```

这里：

- `gridDim` 决定有多少个 block。
- `blockDim` 决定每个 block 有多少个 thread。

硬件执行时，CUDA 会把 block 分配给 SM。

如果 GPU 有很多 SM，那么多个 block 可以同时在不同 SM 上运行。如果 block 数量多于 SM 数量，多出来的 block 会等待前面的 block 完成后再被调度。

这体现了 CUDA 的可扩展性：同一个 kernel 可以在不同 SM 数量的 GPU 上运行，只是并行度不同。

## 4. Warp

在 NVIDIA GPU 上，warp 是线程调度的基本单位。一个 warp 通常包含 32 个线程。

可以通过设备属性查询：

```cpp
devprop.warpSize
```

当前代码打印：

```cpp
printf("warp size: %d\n", devprop.warpSize);
```

一个 block 中的线程会按线程编号线性划分成多个 warp。

例如：

```cpp
blockDim.x = 256
```

那么一个 block 中有：

```text
256 / 32 = 8 warps
```

如果 block 的线程数不是 32 的整数倍，最后一个 warp 也会被分配出来，但会有一些 lane 不工作。因此 block size 通常选择 32 的倍数。

## 5. SIMT 执行模型

CUDA 使用 SIMT，Single Instruction, Multiple Threads。

直观理解：

- 一个 warp 中的线程一起取指令。
- 每个线程有自己的寄存器和线程编号。
- 每个线程可以处理不同数据。
- 同一个 warp 内线程最好走相同控制流。

如果同一个 warp 中的线程遇到分支：

```cpp
if (threadIdx.x < 16) {
  ...
} else {
  ...
}
```

warp 内一部分线程走 `if`，另一部分线程走 `else`，这叫 warp divergence。

发生 divergence 时，硬件会串行执行不同分支，并临时屏蔽不在当前分支中的线程。这样会降低效率。

## 6. SM 上的资源限制

一个 SM 能同时驻留多少个 block，不只取决于 block 数量，还取决于资源。

PMPP 第四章关注的主要资源包括：

- 每个 SM 的 shared memory 容量。
- 每个 SM 的 register 数量。
- 每个 SM 最多 resident blocks。
- 每个 SM 最多 resident threads。
- 每个 block 最多 threads。

这些都可以通过 `cudaDeviceProp` 查询。

当前 `lesson_3.cu` 打印了这些字段：

```cpp
devprop.multiProcessorCount
devprop.sharedMemPerMultiprocessor
devprop.sharedMemPerBlock
devprop.regsPerMultiprocessor
devprop.regsPerBlock
devprop.maxBlocksPerMultiProcessor
devprop.maxThreadsPerMultiProcessor
devprop.maxThreadsPerBlock
```

## 7. 关键设备属性

### SM 数量

```cpp
devprop.multiProcessorCount
```

表示 GPU 中有多少个 SM。

SM 越多，理论上可以同时执行的 block 越多。

当前代码：

```cpp
printf("SM count: %d\n", devprop.multiProcessorCount);
```

### 每个 SM 的 shared memory

```cpp
devprop.sharedMemPerMultiprocessor
```

表示每个 SM 可用的 shared memory 总量。

如果一个 block 使用很多 shared memory，那么同一个 SM 上能同时驻留的 block 数会减少。

例如：

```text
每个 SM shared memory = 100 KB
每个 block 使用 50 KB
最多只能因为 shared memory 驻留 2 个 block
```

### 每个 block 的 shared memory 上限

```cpp
devprop.sharedMemPerBlock
```

表示单个 block 能使用的 shared memory 上限。

如果 kernel 中声明的 shared memory 超过这个值，kernel 不能正常启动。

### 每个 SM 的寄存器数量

```cpp
devprop.regsPerMultiprocessor
```

表示每个 SM 可用的 32-bit register 数量。

每个线程都会使用一定数量的寄存器。如果每个线程用的寄存器太多，同一个 SM 上能驻留的线程数和 block 数都会下降。

### 每个 block 的寄存器数量上限

```cpp
devprop.regsPerBlock
```

表示单个 block 最多可使用的寄存器资源。

寄存器压力太大可能降低 occupancy。

### 每个 SM 最大 resident block 数

```cpp
devprop.maxBlocksPerMultiProcessor
```

表示一个 SM 上最多能同时驻留多少个 block。

注意这是硬件上限，实际能驻留多少还会受到 shared memory、register、线程数共同限制。

### 每个 SM 最大 resident thread 数

```cpp
devprop.maxThreadsPerMultiProcessor
```

表示一个 SM 上最多能同时驻留多少个 thread。

例如如果这个值是 1536，且每个 block 有 256 个线程，那么单从线程数量看：

```text
1536 / 256 = 6 blocks per SM
```

但最终还要看 shared memory、register 和最大 resident block 数。

### 每个 block 最大 thread 数

```cpp
devprop.maxThreadsPerBlock
```

表示单个 block 最多能有多少线程。

常见值是 1024。也就是说：

```cpp
dim3 blockdim(1024);
```

可能是合法的，但：

```cpp
dim3 blockdim(2048);
```

通常不合法。

## 8. Occupancy

Occupancy 通常表示一个 SM 上实际驻留的 warp 数量占最大可驻留 warp 数量的比例。

简化理解：

```text
occupancy = active warps per SM / maximum warps per SM
```

它受这些因素限制：

- block size。
- 每个线程使用的 register 数。
- 每个 block 使用的 shared memory。
- 每个 SM 最大 resident threads。
- 每个 SM 最大 resident blocks。

高 occupancy 的好处是可以帮助隐藏延迟。比如一个 warp 等待 global memory 数据时，SM 可以切换去执行另一个 ready warp。

但 occupancy 不是越高越好。性能还取决于：

- memory access 是否合并。
- arithmetic intensity 是否足够高。
- shared memory 使用是否有效。
- 是否存在 warp divergence。
- 是否有 bank conflict。

## 9. 延迟隐藏

GPU 不是靠单个线程跑得很快，而是靠大量线程隐藏延迟。

当一个 warp 发起 global memory load 后，数据返回可能需要很多周期。SM 不会空等，而是切换执行其他 ready warp。

因此需要足够多的 resident warps。

如果一个 kernel 因为 register 或 shared memory 用太多，导致每个 SM 只能驻留很少的 warps，那么内存延迟就不容易被隐藏，性能可能下降。

## 10. Block Size 选择

选择 block size 时通常考虑：

- 线程数最好是 warp size 的倍数。
- 不超过 `maxThreadsPerBlock`。
- 要让每个 SM 有足够多的 resident warps。
- 不要让单个 block 使用过多 shared memory 或 register。

常见选择：

```cpp
dim3 blockdim(128);
dim3 blockdim(256);
dim3 blockdim(512);
```

对于二维数据，常见选择：

```cpp
dim3 blockdim(16, 16);
dim3 blockdim(32, 8);
```

`16 x 16 = 256` 个线程，正好是 8 个 warp，是图像和矩阵入门代码中常见的配置。

## 11. 当前代码输出的意义

当前 `lesson_3.cu` 输出的属性可以用来判断 kernel 配置是否合理。

例如：

```text
SM count: 24
warp size: 32
shared memory per SM: 102400 bytes
registers per SM: 65536
max resident blocks per SM: 24
max resident threads per SM: 1536
max threads per block: 1024
```

这些信息可以帮助回答：

- 一个 block 最多能开多少线程？
- 一个 SM 理论上最多能同时驻留多少线程？
- 一个 SM 最多能同时驻留多少 block？
- 如果每个 block 使用很多 shared memory，会不会限制并行度？
- 如果每个线程用很多寄存器，会不会降低 occupancy？

## 12. 编译和运行

在仓库根目录执行：

```bash
nvcc 3/lesson_3.cu -o /tmp/lesson_3
/tmp/lesson_3
```

输出示例：

```text
device count: 1

===== device 0 =====
name: NVIDIA GeForce RTX 4060 Laptop GPU
compute capability: 8.9
SM count: 24
warp size: 32
shared memory per SM: 102400 bytes
registers per SM: 65536
max resident blocks per SM: 24
max resident threads per SM: 1536
max threads per block: 1024
```

## 13. 本章小结

PMPP 第四章的核心是从硬件角度理解 CUDA 程序的执行。

需要掌握：

- GPU 由多个 SM 组成。
- block 被调度到 SM 上执行。
- warp 是实际调度单位，通常包含 32 个线程。
- warp divergence 会降低效率。
- SM 的 shared memory、register、最大线程数、最大 block 数都会限制 occupancy。
- occupancy 可以帮助隐藏延迟，但不是唯一性能指标。
- 查询设备属性可以帮助判断 kernel 配置是否合理。

后续做 CUDA 优化时，不能只看代码逻辑是否正确，还要看 kernel 对 SM 资源的使用是否合理。
