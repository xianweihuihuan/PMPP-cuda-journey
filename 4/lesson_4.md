# PMPP 第五章学习笔记

> 第五章标题：**Memory Architecture and Data Locality（内存架构与数据局部性）**。
> 本章笔记按照书里的小节顺序展开（5.1 ~ 5.7），并用 `lesson_4.cu` 里的 tiled 矩阵乘法把每个概念落到代码上。

## 0. 本章主线

第四章讲的是「线程怎么被 SM 调度」，第五章回答一个更现实的问题：**为什么 kernel 已经高度并行了，却还是跑不快？**

书里给出的核心答案是：**朴素 kernel 被 global memory 带宽卡死了**。于是本章的主线是：

```text
访存太多 → 算力利用率极低 → 用片上内存(shared memory)做 tiling → 提高计算访存比 → 逼近峰值算力
```

`lesson_4.cu` 用两个 kernel 演示了这条主线的落地：

```cpp
matrixMulKernel       // 5.4 节：静态 shared memory 的 tiled 矩阵乘法
matrixMulExKernel     // 5.6 节：动态 shared memory（运行时决定 tile 大小）
```

## 1. （5.1）访存效率的重要性

先看最朴素的矩阵乘法，每个线程算一个 `P[Row][Col]`，内层循环是：

```cpp
for (int k = 0; k < width; ++k) {
  val += M[Row * width + k] * N[k * width + Col];
}
```

每次迭代：

- **计算**：1 次乘法 + 1 次加法 = 2 次浮点运算（FLOP）。
- **访存**：从 global memory 读 `M`、`N` 各 1 个 float = 8 字节。

书里用 **计算访存比（compute to global memory access ratio，也叫 arithmetic intensity / 算术强度）** 衡量：

```text
计算访存比 = 浮点运算数 / global memory 访问字节数
           = 2 FLOP / 8 byte = 0.25 FLOP/byte
```

这个 0.25 为什么是灾难？书里用 roofline 思路算了一笔账（以 A100 为例）：

```text
global memory 带宽 ≈ 1555 GB/s
能喂给算力的浮点性能 = 1555 GB/s × 0.25 FLOP/byte ≈ 389 GFLOPS
A100 单精度峰值      ≈ 19500 GFLOPS
利用率 = 389 / 19500 ≈ 2%
```

也就是说，朴素 kernel 把这块顶级 GPU 当成 2% 的算力在用，其余时间全在等内存。

反推：要把峰值算力喂饱，需要的计算访存比是

```text
19500 / 1555 ≈ 12.5 FLOP/byte
```

从 0.25 提升到 12.5，差了约 50 倍。**本章剩下的内容就是想办法把这个比值提上去**，手段就是 tiling。

## 2. （5.2）CUDA 内存类型

要减少 global memory 访问，先得知道 CUDA 有哪些内存可用。书里基于 von Neumann 模型讲了一个关键点：

> 从 **寄存器** 取操作数几乎没有额外开销；而从 **内存** 取操作数需要一条 load 指令、要经过较长延迟、还消耗更多能量。

所以「把反复使用的数据放到离 ALU 更近的地方」本身就是优化。CUDA 的内存层次（书里的 Figure / 变量限定符表）：

| 变量声明 | 所在内存 | 作用域 | 生命周期 |
| --- | --- | --- | --- |
| 自动变量（非数组） | register | thread | kernel |
| 自动数组变量 | local | thread | kernel |
| `__device__ __shared__ int V;` | shared | block | kernel |
| `__device__ int V;` | global | grid | application |
| `__device__ __constant__ int V;` | constant | grid | application |

要点（书里反复强调的）：

- **register**：每个线程私有、片上、最快，但数量有限（第四章说过会限制 occupancy）。
- **shared memory**：**片上**、延迟比 global memory 低一两个数量级，**整个 block 共享**——这正是线程之间协作复用数据的桥梁。
- **global memory**：容量大、片外、延迟高、带宽是瓶颈。
- **constant memory**：只读、有专门缓存，适合所有线程读同一份只读参数。
- **local memory**：名字叫 local，物理上在 global memory 里（寄存器溢出/大数组才会用到），慢。

tiling 的本质就是：**把一块数据从 global memory 搬进 shared memory 一次，让 block 内多个线程反复复用，从而摊薄 global memory 访问。**

## 3. （5.3）用 Tiling 减少访存：拼车类比

书里用了一个很形象的 **拼车（carpool）类比** 来解释 tiling：

- 多个线程要访问的 global memory 数据有大量重叠，就像住在相近地点、要去相近目的地的通勤者。
- 如果各开各的车（每个线程各自去 global memory 取），路上车太多（带宽被打满）。
- 如果**拼车**（大家把共同需要的数据一次性搬进 shared memory 共享），就能大幅减少对道路（global memory 带宽）的占用。
- 但拼车有个前提：**乘客的时间表要接近**——也就是线程访问同一批数据的时间要对得上。如果两个线程一个早上要数据、一个晚上才要，就拼不了车。这对应到代码里就是用 `__syncthreads()` 让大家「步调一致」。

具体到矩阵乘法的数据复用：

- 计算 `P[Row][Col]` 需要 `M` 的第 `Row` 行和 `N` 的第 `Col` 列。
- 同一个 block 内 `TILE_WIDTH × TILE_WIDTH` 个线程，会反复用到 `M`、`N` 中同一批元素。
- 如果各取各的，同一个元素会被重复从 global memory 读很多次。

tiling 的做法是把计算切成若干 **phase（阶段）**，每个 phase 处理一个子块：

1. 把输出矩阵划分成 `TILE_WIDTH × TILE_WIDTH` 的 tile，一个 block 负责一个输出 tile。
2. 每个 phase：block 内线程**协作**把 `M`、`N` 的一个子块搬进 shared memory。
3. 用 shared memory 里的子块做部分累加。
4. 沿 K 方向推进到下一个 phase，重复，直到累加完整行/整列。

这种「把长循环切成一段段子块来做」的技术，书里叫 **strip-mining**。tiling 后每个 global memory 元素只读一次、被 `TILE_WIDTH` 个线程复用，**global memory 流量减少约 `TILE_WIDTH` 倍**，计算访存比也随之提升约 `TILE_WIDTH` 倍。

## 4. （5.4）Tiled 矩阵乘法 kernel

对应 `lesson_4.cu` 的 `matrixMulKernel`，这是本章的核心代码。

### 声明 shared memory 并定坐标

```cpp
#define TILE_WIDTH 16
__shared__ float Msd[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nsd[TILE_WIDTH][TILE_WIDTH];

int tx = threadIdx.x, ty = threadIdx.y;
int bx = blockIdx.x,  by = blockIdx.y;
int Row = by * blockDim.y + ty;   // 该线程负责的输出行
int Col = bx * blockDim.x + tx;   // 该线程负责的输出列
```

`__shared__` 的数组在**编译期**确定大小，整个 block 共用一份。

### phase 循环（strip-mining + 拼车）

```cpp
for (int i = 0; i < width / TILE_WIDTH; ++i) {   // i 就是 phase 编号
  // 1. 协作加载：每个线程负责子块里的一个元素
  Msd[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];
  Nsd[ty][tx] = N[(i * TILE_WIDTH + ty) * width + Col];
  __syncthreads();                 // 屏障①：等所有人都搬完，才能开始算

  // 2. 用 shared memory 里的子块做部分点积
  for (int k = 0; k < TILE_WIDTH; ++k) {
    val += Msd[ty][k] * Nsd[k][tx];
  }
  __syncthreads();                 // 屏障②：等所有人都算完，才能覆盖 shared memory
}
if (Row < width && Col < width) {
  P[Row * width + Col] = val;
}
```

（上面省略了边界判断，完整代码见 5.5 节。）

### 两个 `__syncthreads()`——拼车要「时间表一致」

书里强调这是 tiling 最关键、最容易写错的地方，对应拼车里「乘客时间表要对齐」：

- **屏障①（加载后）**：保证所有线程都把子块写进 shared memory，**再**开始读它做计算。否则会有线程读到别人还没写入的数据。
- **屏障②（计算后）**：保证所有线程都用完当前子块，**再**进入下一个 phase 覆盖 shared memory。否则会有线程在别人还没算完时就把数据改掉了。

书里把这两类同步分别叫 **read-after-write** 和 **write-after-read** 数据依赖的保护。漏掉任何一个都会产生 race condition，结果错误且难复现。

### 复用带来的收益

每个线程在每个 phase 里只从 global memory 读 2 个元素（`Msd`、`Nsd` 各一个），却在内层循环里把它们用了 `TILE_WIDTH` 次。所以 `TILE_WIDTH = 16` 时，global memory 流量降到原来的约 1/16，计算访存比从 0.25 提升到约 4 FLOP/byte。`TILE_WIDTH = 32` 时则约 8 FLOP/byte，对 A100 已能接近 60% 的峰值算力。

## 5. （5.5）边界检查

上面省略的边界判断，正是书里 5.5 节的内容。当 `width` 不是 `TILE_WIDTH` 整数倍时，最后一个 phase 的子块会越界，`lesson_4.cu` 用「越界就填 0」处理：

```cpp
if (Row < width && i * TILE_WIDTH + tx < width) {
  Msd[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];
} else {
  Msd[ty][tx] = 0.0f;          // 越界元素填 0
}
if (i * TILE_WIDTH + ty < width && Col < width) {
  Nsd[ty][tx] = N[(i * TILE_WIDTH + ty) * width + Col];
} else {
  Nsd[ty][tx] = 0.0f;
}
```

为什么填 0 是对的：

- 越界的 `M`/`N` 地址不会被真正访问，避免读非法内存。
- 填进去的 0 参与乘加 = 加 0，不改变最终点积结果。
- 写回时再用 `if (Row < width && Col < width)` 挡一次，保证不写越界的 `P`。

书里特别提醒：加载和写回都要判断边界，缺一不可；而且这个判断不能简单写在 phase 循环外，因为**同一个线程在不同 phase 里可能一会儿有效一会儿越界**。

> 注意：当前 `main` 里 `width == TILE_WIDTH`，`width / TILE_WIDTH` 只有一个 phase、边界恰好对齐，所以填 0 分支不会触发；但 kernel 写成了支持任意尺寸的通用形式。

## 6. （5.6）内存使用对 occupancy 的影响 + 动态 shared memory

书里 5.6 节把第五章和第四章接上了：**shared memory 既能减少访存，本身又是限制并行度的资源。**

- 每个 block 用的 shared memory 越多，一个 SM 上能同时驻留的 block 就越少，occupancy 下降。
- 所以 `TILE_WIDTH` 是一个权衡：太小复用不足、计算访存比提不上去；太大单 block 占用 shared memory 过多、occupancy 掉下来。

更重要的是，书里指出：**不同 GPU 每个 block 可用的 shared memory 不一样**，写死的 `TILE_WIDTH` 不能适配所有设备。理想做法是**运行时查询设备属性**（正是 `lesson_3.cu` 做的 `cudaGetDeviceProperties` → `sharedMemPerBlock`），再据此决定 tile 大小。可一旦 tile 大小要到运行时才确定，`__shared__ float a[TILE_WIDTH][TILE_WIDTH]` 这种编译期写法就不够用了，于是需要**动态 shared memory**。

这就是 `matrixMulExKernel` 存在的意义。

### 在 kernel 里：用 `extern __shared__` 声明、手动切分

```cpp
extern __shared__ float Msd_Nsd[];          // 没有大小，运行时再定
float* Msd = (float*)Msd_Nsd;
float* Nsd = (float*)Msd_Nsd + size;        // 两块共用同一缓冲区，手动错开
```

- 一个 kernel 只能有**一块** `extern __shared__` 缓冲区，所以 `Msd`、`Nsd` 要手动切分。
- `size` 是单个子块元素数（`TILE_WIDTH * TILE_WIDTH`），`Nsd` 从 `Msd` 之后开始。
- 因为是一维缓冲区，二维下标要手动展开：`Msd[ty * blockDim.x + tx]`。

### 在 host 端：用第三个 `<<<>>>` 参数指定字节数

```cpp
int sharedElements = TILE_WIDTH * TILE_WIDTH;
int sharedBytes    = 2 * sharedElements * sizeof(float);   // M、N 两块都要算进去

matrixMulExKernel<<<griddim, blockdim, sharedBytes>>>(
    M_d, N_d, P_d, width, sharedElements);
```

`<<<grid, block, sharedBytes>>>` 的第三个参数是**动态 shared memory 的字节数**，它会在启动时分配。

### 静态 vs 动态 对比

| | 静态 shared memory | 动态 shared memory |
| --- | --- | --- |
| 大小确定时机 | 编译期 | kernel 启动时 |
| 声明 | `__shared__ float a[N][N];` | `extern __shared__ float a[];` |
| 多个数组 | 直接声明多个 | 手动切分同一块缓冲区 |
| 下标 | 编译器算二维 | 手动展开成一维 |
| 适用场景 | tile 大小固定 | tile 大小要按设备/运行时决定 |

两个 kernel 计算逻辑完全相同，区别只在 shared memory 怎么分配。


## 7. （5.7）本章小结

PMPP 第五章的主线：**算力很贵的不是计算，而是访存——用片上内存做 tiling 把数据局部性榨出来。**

- §5.1 朴素矩阵乘法计算访存比只有 0.25 FLOP/byte，A100 上利用率约 2%，被 global memory 带宽卡死。
- §5.2 CUDA 内存层次：register / shared / global / constant / local；shared memory 片上、block 内共享，是协作复用的关键。
- §5.3 tiling 像拼车——共享数据减少访存，但要求线程「时间表一致」，靠 `__syncthreads()` 实现；技术上是 strip-mining，把计算切成多个 phase。
- §5.4 tiled 矩阵乘法 kernel：协作加载子块 → 屏障① → 部分点积 → 屏障②；global memory 流量降为 1/`TILE_WIDTH`。
- §5.5 边界检查：越界填 0，加载和写回都要判断。
- §5.6 shared memory 也是受限资源，会压低 occupancy；要适配不同设备就用运行时查询 + 动态 shared memory（`extern __shared__` + 第三个启动参数）。

后续的矩阵乘法优化（寄存器分块、向量化访存、消除 bank conflict）都建立在本章的 tiling 与数据局部性思想之上。
