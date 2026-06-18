# PMPP 第七章学习笔记

> 第七章标题：**Convolution（卷积）**。
> 本章是第一个「真实计算模式（stencil/卷积）」案例，把前几章的 constant memory、tiling、halo 处理综合到一起。笔记按书的小节顺序展开（7.1 ~ 7.6），并用 `lesson_6.cu` 里 5 个逐步优化的 kernel 把概念落到代码上。

## 0. 本章主线

卷积（convolution）是图像处理、信号处理、CNN 的基础算子：用一个小的 **filter（滤波器，也叫 convolution kernel / mask）** 在输入上滑动，每个输出是邻域的加权和。

书里用卷积演示一条完整的优化链，`lesson_6.cu` 的 5 个 kernel 正好对应这条链：

```cpp
convolution_1D_Kernel                      // 7.2 基本并行卷积（1D）
convolution_2D_Kernel                      // 7.2 基本并行卷积（2D，filter 在 global memory）
convolution_2D_Kernel_by_const             // 7.3 把 filter 放进 constant memory
convolution_2D_kernel_by_const_shared1     // 7.4 tiled 卷积 + halo cells（输入 tile 比输出 tile 大）
convolution_2D_kernel_by_const_shared      // 7.5 tiled 卷积 + 用 cache 处理 halo
```

## 1. （7.1）背景：卷积是什么

一维卷积：输出 `P[i]` 是以 `i` 为中心、半径 `r` 的邻域加权和：

```text
P[i] = Σ_{j=0}^{2r+1-1}  M[i - r + j] * F[j]
```

- `F` 是长度 `2r+1` 的 filter。
- `r` 是 **filter radius（半径）**，filter 大小 = `2r+1`（1D）或 `(2r+1)×(2r+1)`（2D）。

二维卷积就是在行、列两个方向都做加权和。

### 两类边界元素（术语要分清）

书里特别区分了两个容易混的概念，后面优化全靠它：

- **ghost cells（幽灵元素）**：滑窗超出**整个输入数组**边界的位置。通常按 0 处理（zero padding）。
- **halo cells（晕圈元素）**：做 tiling 时，一个 tile 计算输出需要用到**邻居 tile 的边缘输入**。它们不在边界外，只是不属于本 tile 的输出区域。

> 命名提醒：书里把滤波器叫 filter / convolution kernel / mask，是为了和 CUDA 的 kernel 区分。本仓库代码里它就是 `F`。

## 2. （7.2）基本并行卷积

### 1D：`convolution_1D_Kernel`

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < l) {
  float val = 0.0f;
  for (int j = 0; j < 2 * r + 1; ++j) {
    int idx = i - r + j;
    if (idx >= 0 && idx < l) {     // ghost cell：越界就跳过（等价于乘 0）
      val += M[idx] * F[j];
    }
  }
  P[i] = val;
}
```

一个线程算一个输出 `P[i]`，循环遍历 filter，`if (idx >= 0 && idx < l)` 就是 **ghost cell 的边界处理**：越界的输入不参与，等价于按 0 padding。

### 2D：`convolution_2D_Kernel`

```cpp
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;
...
for (int frow = 0; frow < 2*r+1; ++frow)
  for (int fcol = 0; fcol < 2*r+1; ++fcol) {
    int inrow = Row - r + frow;
    int incol = Col - r + fcol;
    if (inrow >= 0 && inrow < width && incol >= 0 && incol < width)
      val += M[inrow * width + incol] * F[frow * (2*r+1) + fcol];
  }
```

逻辑一样，只是双层循环 + 二维边界判断。

### 为什么需要优化

每个输出要做 `(2r+1)²` 次乘加，同时访问 `M` 和 `F` 各 `(2r+1)²` 次。**计算访存比依然很低**，而且：

- `F` 被**每个线程**反复读，但它对所有线程都一样、且只读不变 → 适合 constant memory（7.3）。
- 相邻输出的输入邻域大量重叠 → 适合 tiling（7.4 / 7.5）。

## 3. （7.3）Constant Memory 与缓存

这是第七章里第一次系统讲缓存，也是本仓库第一次用到 constant memory，值得展开。

### 3.1 为什么 filter 适合放进 constant memory

书里给出三条判断标准，filter `F` 全中：

1. **小**：卷积 mask 一般就几十到几百个元素（constant memory 总共约 64 KB）。
2. **只读 / kernel 执行期间不变**：没有任何线程会写它。
3. **所有线程访问同一批元素、且访问顺序相同**：从 `F[0][0]` 开始按相同次序遍历。

> 「`M`(filter) is a good candidate for Constant Memory」——这三条就是书里反复强调的标准。

### 3.2 怎么用（两步）

`convolution_2D_Kernel_by_const` 把 `F` 改成 `__constant__`：

```cpp
#define FILTER_RADIUS 2
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];   // ① 文件作用域声明

__global__ void convolution_2D_Kernel_by_const(float* M, float* P, int width) {
  ...
  val += M[inrow * width + incol] * F[frow][fcol];   // F 直接从 constant memory 读
}
```

```cpp
// ② host 端用「符号名」拷贝，而不是指针
cudaMemcpyToSymbol(F, h_F, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
```

注意三点：

1. `__constant__` 变量是**文件作用域**声明（不能在函数内），不再作为 kernel 参数传入（kernel 签名里 `F` 消失了）。
2. host 端用 `cudaMemcpyToSymbol(符号, 源指针, 字节数)` 拷数据（`lesson_6.cu` 只有 kernel，运行时需要补这步）。这个关键字本质上是**告诉 GPU「缓存这块数据是安全的」**。
3. `FILTER_RADIUS` 变成编译期常量，循环边界固定，编译器能更激进地展开优化。

### 3.3 缓存（cache）到底是什么

书在这里第一次正式介绍缓存：

- **缓存按「行（line）」存储**：一条 cache line ≈ 一次 DRAM burst ≈ 128 字节。一次内存读取产生一整行，缓存存这行副本，并用 **tag** 记录它对应的内存地址。
- **缓存比内存小**，装不下全部数据；满了按 **LRU（最近最少使用）** 淘汰旧行。
- 缓存同时吃**空间局部性**和**时间局部性**——所以遍历 2D mask 时按存储顺序访问更利于命中。
- **延迟层次**（数量级）：寄存器最快 → shared memory / L1 → **constant memory 带缓存时约 5 个周期** → 不命中回退到 global memory（几百周期）。

### 3.4 常量缓存 vs 共享内存（本节最关键的对比）

| | 常量缓存（cache） | 共享内存（scratchpad） |
| --- | --- | --- |
| 谁决定内容 | **微架构自动决定**（透明） | **程序员显式管理** |
| 读写 | 只读 | 可读可写 |
| 物理介质 | 片上 SRAM | 片上 SRAM（Volta 起与 L1 共用物理资源、动态分配） |
| 性能 | 相近 | 相近 |

书里的关键洞察——**为什么只读的常量缓存能比普通 L1 更快**：

- 普通 L1 要支持写，就必须跟踪每行的修改状态、把改动写回内存、维护跨核**一致性（coherence）**。
- 常量 / 纹理缓存的行是**只读**的 → 不需要写回、不需要维护一致性 → 因此能做得**更激进、吞吐更高、更省电**。
- 再加上 **broadcast（广播）**：当一个 warp 里所有线程读**同一个**常量地址时，硬件一次把值广播给所有线程。这正是卷积里每个线程读同一个 `F[frow][fcol]` 的模式。

可以记成一条因果链：**只读 → 无需一致性 → 缓存可以更激进 + 可广播 → 访问近乎免费。**

### 3.5 对性能的实际意义：计算访存比翻倍

`F` 进常量缓存后基本不占 DRAM 带宽，于是算「计算访存比」时 global memory 访问**只需数输入 `M`**：

```text
基本卷积： 每次乘加读 M 和 F 各一个 → 2 FLOP / 8 byte = 0.25 FLOP/byte
F 进常量缓存： 只剩 M 的 4 字节  → 2 FLOP / 4 byte = 0.50 FLOP/byte（翻倍）
```

而且常量缓存**不占用 SM 的普通 L1 / shared memory 预算**。优化后主要的 global memory 流量只剩对输入 `M` 的访问——下一步就拿 tiling 来削它。

## 4. （7.4）Tiled 卷积 + Halo Cells：`convolution_2D_kernel_by_const_shared1`

核心思路：把一个 tile 要用到的输入（**包括 halo**）一次性搬进 shared memory，之后卷积全部从 shared memory 读。

关键尺寸关系——**输入 tile 比输出 tile 大一圈 halo**：

```cpp
#define IN_TILE_DIM  32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)   // 32 - 4 = 28
```

- block 大小 = `IN_TILE_DIM × IN_TILE_DIM`（32×32），负责**加载**整个输入 tile（含两侧各 `r` 的 halo）。
- 但只有中间 `OUT_TILE_DIM × OUT_TILE_DIM`（28×28）个线程**产出**输出。

### 加载（含 halo 和 ghost）

```cpp
int col = bx * OUT_TILE_DIM + tx - FILTER_RADIUS;   // 注意按 OUT_TILE_DIM 步进，并左移 r
int row = by * OUT_TILE_DIM + ty - FILTER_RADIUS;
if (row >= 0 && row < width && col >= 0 && col < width)
  N_s[ty][tx] = N[row * width + col];
else
  N_s[ty][tx] = 0.0f;                                // ghost cell 填 0
__syncthreads();
```

- 相邻 block 的输入 tile 在 grid 上按 `OUT_TILE_DIM` 步进，所以相邻 tile 的 halo 是**重叠**的——这正是被复用的部分。
- `-FILTER_RADIUS` 让加载范围向左上扩出一圈 halo。
- 越界处填 0（ghost）。

### 计算（只让内部线程产出）

```cpp
int tileCol = tx - FILTER_RADIUS;
int tileRow = ty - FILTER_RADIUS;
if (tileCol >= 0 && tileCol < OUT_TILE_DIM &&
    tileRow >= 0 && tileRow < OUT_TILE_DIM) {       // 只有内部 28×28 线程算
  float val = 0.0f;
  for (int frow = 0; frow < 2*FILTER_RADIUS+1; ++frow)
    for (int fcol = 0; fcol < 2*FILTER_RADIUS+1; ++fcol)
      val += N_s[ty - FILTER_RADIUS + frow][tx - FILTER_RADIUS + fcol] * F_c1[frow][fcol];
  P[row * width + col] = val;                       // 卷积全程只读 shared memory
}
```

### 代价：边缘线程被浪费

这种方案的缺点（书里明确指出）：负责加载 halo 的那一圈线程**只搬数据、不算输出**。利用率 = `(OUT_TILE_DIM / IN_TILE_DIM)²`：

```text
(28 / 32)² ≈ 0.77
```

filter 越大、tile 越小，halo 占比越高，浪费越严重。于是有了 7.5 的改进。

## 5. （7.5）Tiled 卷积 + 用 Cache 处理 Halo：`convolution_2D_kernel_by_const_shared`

改进思路：**让 block 大小 = 输出 tile 大小**，shared memory 只装本 tile 的「内部」元素；halo 元素需要时直接从 global memory 读，**赌它们已经被邻居 block 加载、命中 L2 cache**。

```cpp
#define TILE_DIM 32
int Col = bx * TILE_DIM + tx;       // 没有 -r 偏移，block 直接对齐输出 tile
int Row = by * TILE_DIM + ty;
if (Row < width && Col < width)
  N_s[ty][tx] = N[Row * width + Col];   // 只加载内部元素，不含 halo
else
  N_s[ty][tx] = 0.0f;
__syncthreads();
```

卷积时对每个邻居分两种情况：

```cpp
if (ty - r + frow ∈ [0,TILE_DIM) && tx - r + fcol ∈ [0,TILE_DIM)) {
  val += N_s[ty - r + frow][tx - r + fcol] * F_c2[frow][fcol];   // 在本 tile 内 → 读 shared
} else if ( ...邻居在 width 范围内... ) {
  val += N[(Row - r + frow) * width + (Col - r + fcol)] * F_c2[frow][fcol];  // halo → 读 global，靠 cache
}
```

（`r` 即 `FILTER_RADIUS`，上面为简洁缩写。）

对比 7.4：

| | 7.4 halo cells（`shared1`） | 7.5 cache halo（`shared`） |
| --- | --- | --- |
| block 大小 | 输入 tile（含 halo，偏大） | 输出 tile（=`TILE_DIM`） |
| shared memory 内容 | 内部 + halo | 只有内部 |
| halo 来源 | 全在 shared memory | 直接读 global，靠 L2 cache |
| 线程利用率 | 边缘线程被浪费 | 所有线程都算输出 |
| 代码复杂度 | 索引偏移绕 | 多一个 else 分支判断 |

书里的观点：现代 GPU 有较大的 L2 cache，halo 元素被邻居 block 读过后大概率仍在 cache 里，所以「让 cache 兜底 halo」往往比「专门搬 halo」更简洁、利用率更高。

## 6. 验证与编译

`lesson_6.cu` 目前只有 kernel，要跑起来需要补 host 主程序：用 `cudaMemcpyToSymbol` 给各个 `__constant__` filter 赋值，再写一个朴素 CPU 卷积对拍。注意几个 grid 配置差异：

- `convolution_2D_kernel_by_const_shared1`：block = `IN_TILE_DIM`(32)，grid 按 **OUT_TILE_DIM**(28) 划分。
- `convolution_2D_kernel_by_const_shared`：block = grid 都按 `TILE_DIM`(32) 划分。

```bash
nvcc 6/lesson_6.cu -o /tmp/lesson_6   # 需要先补 main + cudaMemcpyToSymbol
```

## 7. （7.6）本章小结

PMPP 第七章用卷积串起了一条完整优化链：

- §7.1 卷积 = filter 滑窗加权和；分清 **ghost cells**（越界，填 0）和 **halo cells**（邻居 tile 的边缘）。
- §7.2 基本 kernel：一个线程一个输出，边界用 if 判断；计算访存比低。
- §7.3 filter 只读且小且人人共用 → 放进 **constant memory**（`__constant__` + `cudaMemcpyToSymbol`），命中 constant cache、不占 shared 预算、半径变编译期常量。
- §7.4 **tiled + halo cells**：输入 tile 比输出 tile 大一圈，halo 全进 shared memory；缺点是边缘线程只加载不计算，利用率 `(OUT/IN)²`。
- §7.5 **tiled + cache halo**：block 对齐输出 tile，只缓存内部元素，halo 直接读 global memory 靠 L2 cache 兜底；利用率更高、代码更简洁。

一句话：**卷积优化 = 不变的 filter 进 constant memory + 重叠的输入靠 tiling/cache 复用 + 小心 ghost/halo 边界。**
