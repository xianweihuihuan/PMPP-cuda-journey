# PMPP 第六章学习笔记

> 第六章标题：**Performance Considerations（性能考量）**。
> 本章把前几章的「能跑对」推进到「跑得快」，按书里的小节顺序展开（6.1 ~ 6.6），并用 `lesson_5.cu` 的线程粗化（thread coarsening）矩阵乘法、`test_1.cu` 的合并访问实验把概念落到代码上。

## 0. 本章主线

第五章用 tiling 把矩阵乘法的 global memory 流量降了下来。第六章继续问：**还能再快吗？瓶颈到底在哪？**

书里把性能优化拆成几条相对独立的「旋钮」，每条对应一节：

```text
6.1 内存合并访问 (memory coalescing)    —— 让一个 warp 的访存合并成少数几次 DRAM 传输
6.2 隐藏内存延迟 (latency hiding)        —— 靠 DRAM 的 bank/channel 并行 + 足够多的 warp
6.3 线程粗化 (thread coarsening)         —— 一个线程多干活，省掉并行带来的冗余开销
6.4 优化清单                              —— 把上面这些旋钮汇总成 checklist
6.5 找瓶颈                                —— 先判断是 compute-bound 还是 memory-bound 再优化
```

`lesson_5.cu` 重点演示了 **6.3 线程粗化**：

```cpp
matrixMulKernel     // 1D 粗化：一个线程算 COARSE_FACTOR 个相邻列方向的输出
matrixMul2DKernel   // 2D 粗化：一个线程算 COARSE_FACTOR × COARSE_FACTOR 个输出 tile
```

## 1. （6.1）内存合并访问（Memory Coalescing）

### DRAM 是「成批」读的

书里先讲硬件事实：global memory（DRAM）一次访问不是取 1 个字节，而是取**一整段连续地址（burst）**。所以：

> 如果一个 warp 里 32 个线程访问的是**连续**的 global memory 地址，硬件可以把它们合并（coalesce）成少数几次 DRAM 传输；如果地址东一个西一个，就要拆成很多次传输，带宽被浪费。

判断是否合并的关键：**看 warp 内相邻线程（相邻 `threadIdx.x`）访问的地址是不是相邻**。

### 落到 tile 加载上

行主序存储下，`lesson_5.cu` 的加载是合并的。以 M 为例：

```cpp
MDs[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];
```

相邻线程的 `tx` 相差 1，访问的 M 地址也正好相差 1（`+ tx`）→ 一个 warp 的读请求落在连续地址上 → **合并访问**。N 的加载同理（`Col = ColStart + ... + tx`，也是 `+tx`）。

这就是为什么 tile 加载要让 `tx` 落在「最内层连续维度」上——不是随便写都能合并的。

### `test_1.cu`：corner turning（转角）

`test_1.cu` 演示了书里讲的一个技巧——当某个矩阵的自然访问方向会导致**不合并**时，可以让 **global 端按连续地址读、在 shared memory 端转置存放**：

```cpp
NDs[tx][ty] = N[(bx * TILE_WIDTH + ty) * width + i * TILE_WIDTH + tx];
//  ^转置存                                                    ^仍然 +tx，连续读
```

global 读仍然是 `+tx` 连续（合并），写进 shared memory 时换成 `NDs[tx][ty]` 做转置。shared memory 是片上的，转置带来的非连续访问代价远小于 global memory，这就是 **corner turning**：把不合并的访问从 global memory「挪到」shared memory 上承担。

## 2. （6.2）隐藏内存延迟（Latency Hiding）

即使访问合并了，DRAM 延迟依然很高。书里解释 GPU 怎么把延迟藏起来：

- DRAM 内部分成多个 **bank** 和 **channel**，不同 bank/channel 可以**并行**服务不同请求。数据在物理上是交错（interleaved）分布到各 channel 的，所以连续的大块访问能同时压满多个 channel。
- 光有并行通道还不够，还需要**足够多的 in-flight 访存请求**去喂满它们——这正是第四章说的：一个 SM 上要驻留足够多的 warp，当一个 warp 在等数据时，调度器切到另一个 ready warp，从而把延迟藏在计算后面。

结论：**合并访问（6.1）决定单次传输效率，足够的 occupancy + bank/channel 并行（6.2）决定能不能把延迟藏住**，两者配合才能逼近带宽上限。

## 3. （6.3）线程粗化（Thread Coarsening）—— 本章核心代码

### 并行不是免费的

前几章的思路都是「线程开得越细越好」（一个线程算一个输出）。书里在这里点出代价：

> 把任务切到最细粒度，对**可扩展性**好（硬件资源多就并行多）；但如果硬件本来就要把一部分 block **串行**执行，那么这些「本可以共享、却被拆到不同线程/block 各做一遍」的冗余工作就白白多花了。

矩阵乘法里典型的冗余是：**计算相邻的输出 tile 时，会重复加载同一块输入 tile**。如果这些相邻 tile 被分给不同 block，各自从 global memory 把同一块数据再读一遍，就是浪费。

**线程粗化**的做法：让一个线程（一个 block）负责**多个相邻输出**，把本来要重复加载的输入 tile **只加载一次、复用多次**。

### 1D 粗化：`matrixMulKernel`

每个线程负责**同一行、COARSE_FACTOR 个相邻列**方向的输出：

```cpp
#define TILE_WIDTH 16
#define COARSE_FACTOR 4

int Row      = by * TILE_WIDTH + ty;
int ColStart = bx * (TILE_WIDTH * COARSE_FACTOR) + tx;   // 一个 block 横向覆盖 4 个 tile
float vals[COARSE_FACTOR] = {0};                          // 4 个累加器

for (int i = 0; i < width / TILE_WIDTH; ++i) {
  MDs[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];     // ★ M tile 每个 phase 只加载一次
  for (int c = 0; c < COARSE_FACTOR; ++c) {
    int Col = ColStart + c * TILE_WIDTH;
    NDs[ty][tx] = N[(i * TILE_WIDTH + ty) * width + Col]; // N tile 每个列各加载一次
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k) {
      vals[c] += MDs[ty][k] * NDs[k][tx];                 // 复用同一份 MDs
    }
    __syncthreads();
  }
}
```

关键在那行 `★`：**M tile 在 `c` 循环外只加载一次**，却被 4 个不同的 N tile（4 个输出列）复用。对比第五章的非粗化版本——那里每个输出列各自属于不同线程，会把 M tile 各自重读一遍。粗化把这部分 global memory 冗余访问省掉了。

> ✅ 已修复的 bug：写回原本写成 `P[Row * width + ColStart + k * width]`，但 `vals[k]` 是按**列**（`Col = ColStart + k * TILE_WIDTH`）累加出来的，`k * width` 会沿行方向乱写。正确写法是：
> ```cpp
> P[Row * width + ColStart + k * TILE_WIDTH] = vals[k];
> ```
> 这个 bug 是补了 main + CPU 对拍后才暴露的（见 §6）——印证了「优化 kernel 必须对拍」。

### 2D 粗化：`matrixMul2DKernel`

更进一步，每个线程负责 **COARSE_FACTOR × COARSE_FACTOR 个输出 tile**，M、N 两个方向的输入都能复用：

```cpp
__shared__ float MDs[TILE_WIDTH][TILE_WIDTH];
__shared__ float NDs[COARSE_FACTOR][TILE_WIDTH][TILE_WIDTH];   // 一次缓存 4 个 N tile
float vals[COARSE_FACTOR][COARSE_FACTOR] = {0};                // 4×4 个累加器

for (int i = 0; i < width / TILE_WIDTH; ++i) {
  for (int c = 0; c < COARSE_FACTOR; ++c)                      // 先把这一 phase 的 4 个 N tile 都装进来
    NDs[c][ty][tx] = N[(ty + i*TILE_WIDTH)*width + ColStart + c*TILE_WIDTH];  // 注意 ColStart 已含 +tx

  for (int r = 0; r < COARSE_FACTOR; ++r) {
    int Row = RowStart + r * TILE_WIDTH;
    MDs[ty][tx] = M[Row * width + i * TILE_WIDTH + tx];        // 每行的 M tile 加载一次
    __syncthreads();
    for (int c = 0; c < COARSE_FACTOR; ++c)                    // 用这一份 MDs 配 4 个 NDs
      for (int k = 0; k < TILE_WIDTH; ++k)
        vals[r][c] += MDs[ty][k] * NDs[c][k][tx];
    __syncthreads();
  }
}
```

> ✅ 已修复的 bug：N tile 加载原本写成 `... + ColStart + c*TILE_WIDTH + tx`，但 `ColStart` 已经含 `+tx`，再加一次会让列号偏移翻倍、读错列。正确写法去掉末尾的 `+ tx`（即上面代码的样子）。这个 bug 同样是 CPU 对拍才发现的（修复前 `2D max error` 高达 2948）。

复用关系：

- 每个 phase 的 4 个 N tile 缓存进 `NDs[COARSE_FACTOR][..][..]`，被 4 行（`r`）共用。
- 每行的 M tile 加载一次，被 4 列（`c`）共用。
- 于是一个线程用 `4 + 4` 次 tile 加载，算出了 `4 × 4 = 16` 个输出 tile 的贡献——**计算访存比进一步提升**。

### 粗化不是越多越好

书里强调的副作用（写笔记一定要记）：

- 每个线程用更多 **寄存器**（`vals[]` 累加器）和更多 **shared memory**（2D 版的 `NDs` 大了 COARSE_FACTOR 倍）→ 压低 occupancy。
- 粗化倍数太大 → 总线程数变少 → 可能喂不满硬件、藏不住延迟。
- 所以 `COARSE_FACTOR` 是个权衡值，要针对具体 GPU 实测，不能盲目调大。

## 4. （6.4）优化清单

书里把全书的优化手段汇成一张 checklist，对应到本仓库已经写过的代码：

| 优化点 | 含义 | 对应代码 |
| --- | --- | --- |
| 最大化 occupancy | 让 SM 驻留足够多 warp 来藏延迟 | 第四章设备属性查询 |
| 合并访问 | warp 内连续地址 | §6.1，`+tx` 索引、`test_1.cu` corner turning |
| 减少控制分歧 | 同 warp 走相同分支 | 第四章 warp divergence |
| tiling 提升局部性 | 把数据搬进 shared memory 复用 | 第五章 tiled matmul |
| 线程粗化 | 一个线程多算，省冗余访存 | 本章 `lesson_5.cu` |

## 5. （6.5）先找瓶颈再优化

书里最后的方法论：**不要盲目套优化，先判断 kernel 是 compute-bound 还是 memory-bound。**

- **memory-bound**（被访存卡住）：合并访问、tiling、线程粗化这类「减少/优化访存」的手段才有效——矩阵乘法的朴素版正是这种。
- **compute-bound**（被算力卡住）：再优化访存收益不大，要去减少指令数、降低精度、用更高吞吐的指令。

判断手段是 profiler（如 Nsight Compute），看实测带宽利用率、计算单元利用率、occupancy 等指标，对照 roofline（第五章那条线）定位瓶颈，再决定动哪个旋钮。**先测量，后优化**。

## 6. 验证与编译

矩阵乘法的优化 kernel 很容易写错下标，`lesson_5.cu` 已经补上 `main` + 朴素 CPU 对拍（`matrixMulCpu` / `maxError`），用 `width = 128`（必须是 `TILE_WIDTH * COARSE_FACTOR = 64` 的倍数，两个 kernel 都没有边界检查）：

```bash
nvcc -arch=sm_89 5/lesson_5.cu -o /tmp/lesson_5
/tmp/lesson_5
```

修复两个下标 bug 后的输出：

```text
width = 128
1D coarsening max error: 0
2D coarsening max error: 0
```

**这一节最大的收获不是粗化本身，而是：两个 bug 单看代码都不明显，是 CPU 对拍把它们逼出来的。优化 kernel 必须对拍。**

## 7. （6.6）本章小结

PMPP 第六章是「性能旋钮合集」：

- §6.1 合并访问：warp 内相邻线程访问相邻地址，才能把访存合并成少数 DRAM 传输；不合并时可用 corner turning 把代价挪到 shared memory（`test_1.cu`）。
- §6.2 隐藏延迟：靠 DRAM 的 bank/channel 并行 + 足够多的驻留 warp，把高延迟藏在计算后面。
- §6.3 线程粗化：并行有冗余代价；让一个线程算多个相邻输出，把可复用的输入 tile 只加载一次（`lesson_5.cu` 的 1D / 2D 粗化），但会增加寄存器/shared memory 占用、压低 occupancy，需权衡。
- §6.4 把以上汇成优化 checklist。
- §6.5 先用 profiler 判断 compute-bound / memory-bound，再决定优化方向。

一句话：**性能优化不是堆技巧，而是「定位瓶颈 → 选对旋钮 → 实测验证」的循环。**
