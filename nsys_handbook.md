# Nsight Systems (`nsys`) 命令行参考手册

> 环境：Nsight Systems **2025.6** / RTX 4060 Laptop (sm_89) / CUDA `/usr/local/cuda`。
> 本文档逐子命令、逐选项解释 `nsys --help` 的含义、取值、默认值、用法和输出结果含义，配例子。
> 命令示例统一用本仓库的程序（`/tmp/lesson_5`、`/tmp/lesson_6`）。

---

## 1. 这个工具是干什么的

`nsys` 是 **整个程序的时间线分析器**：在毫秒级的全局视角看 CPU 与 GPU 在每个时刻干什么——kernel 怎么排布、memcpy 占了多少、CUDA API 在哪等待、流之间是否并发、CPU 有没有把 GPU 喂满。

和 `ncu` 的分工：

| | `nsys`（本文） | `ncu` |
| --- | --- | --- |
| 看什么 | 全程序时间线（宏观） | 单 kernel 内部计数器（微观） |
| 开销 | 很低（采样/追踪） | 很高（kernel 重放多遍） |
| 顺序 | **先用它**定位「时间花在哪、哪个 kernel 热」 | nsys 定位后再钻进去 |

**正确流程：先 nsys 看全局 → 找到热点 kernel → 再 ncu 深挖。**

---

## 2. 命令结构：nsys 是「子命令」式

```
nsys [--version] [--help] <command> [<args>] [application] [<application args>]
```

`nsys --help` 列出的常用子命令：

| 子命令 | 作用 |
| --- | --- |
| `profile` | **跑一遍程序并采集时间线**，产物 `.nsys-rep`（最常用） |
| `stats` | 从已有 `.nsys-rep`/`.sqlite` 生成**文字统计表** |
| `analyze` | 从报告里自动找优化点 |
| `launch` / `start` / `stop` / `cancel` | 交互式分段采集（先 launch，再 start/stop 圈定） |
| `export` | 把 `.nsys-rep` 导成其它格式（如 sqlite） |
| `status` | 查看 CLI / 采集环境状态 |
| `stats`/`analyze` 用法 | `nsys <command> --help` 查各自帮助 |

---

## 3. `nsys profile` —— 采集时间线

```
nsys profile [<args>] <application> [<app args>]
```

### 3.1 最常用选项

| 选项 | 取值 / 默认 | 含义 | 例子 |
| --- | --- | --- | --- |
| `-o, --output` | 文件名 | 报告输出名（自动加 `.nsys-rep`） | `-o /tmp/report` |
| `-f, --force-overwrite` | `true`/`false`，默认 `false` | 覆盖同名报告 | `-f true` |
| `-t, --trace` | 见下，默认 `cuda,nvtx,osrt,opengl` | **追踪哪些子系统**，逗号分隔 | `-t cuda,nvtx` |
| `--stats` | `true`/`false`，默认 `false` | 跑完直接在终端打印统计（等于自动跑 `nsys stats`） | `--stats=true` |
| `-d, --duration` | 秒，默认 `0`（不限） | 最多采集多少秒 | `-d 5` |
| `-s, --sample` | `process-tree`/`system-wide`/`none` | CPU 采样范围；`none` 进一步降开销 | `-s none` |
| `-e, --env-var` | `A=B,C=D` | 给目标程序设环境变量 | `-e CUDA_VISIBLE_DEVICES=0` |
| `-w, --show-output` | `true`/`false` | 是否把目标程序的 stdout/stderr 透传到终端 | |

`-t / --trace` 取值：

| 值 | 含义 |
| --- | --- |
| `cuda` | CUDA API + kernel + memcpy（**核心**） |
| `nvtx` | 代码里手动打的 NVTX 区间（见 §6） |
| `osrt` | OS 运行时：线程调度、系统调用、CPU 等待 |
| `cublas` / `cudnn` / `cusparse` … | 对应库的调用 |
| `opengl` / `vulkan` | 图形 API |
| `none` | 不追踪该类 |

### 3.2 圈定采集范围（减少噪音）

| 选项 | 取值 / 默认 | 含义 |
| --- | --- | --- |
| `-c, --capture-range` | `none`(默认) / `cudaProfilerApi` / `nvtx` / `hotkey` | 何时**开始**采集。`cudaProfilerApi`=程序调用 `cudaProfilerStart()` 才开始；`nvtx`=进入指定 NVTX 区间才开始 |
| `--capture-range-end` | `none`/`stop`/`stop-shutdown`(默认)/`repeat[:N]` | 范围结束时的行为：停止采集 / 停止并关会话 / 重复采集 N 次 |
| `--delay` | 秒 | 启动后延迟多少秒再开始采集 |

> `-c cudaProfilerApi` 要求同时开 CUDA 追踪（`-t cuda`），并在代码里 `#include <cuda_profiler_api.h>` 调用 `cudaProfilerStart/Stop`。

### 3.3 进阶/低频选项

| 选项 | 取值 / 默认 | 含义 |
| --- | --- | --- |
| `--cuda-memory-usage` | `true`/`false`，默认 `false` | 记录 GPU 显存使用（**有明显开销**） |
| `--cuda-um-cpu-page-faults` / `--cuda-um-gpu-page-faults` | `true`/`false` | 记录统一内存的 CPU/GPU 缺页（**开销大**） |
| `--cuda-graph-trace` | `graph`(默认,驱动≥11.7) / `node` | CUDA Graph 按整图还是按节点追踪 |
| `-b, --backtrace` | `lbr`(默认)/`fp`/`dwarf`/`none` | CPU 采样时的回溯方式 |
| `--cpuctxsw` | `process-tree`/`system-wide`/`none` | 是否追踪 CPU 线程上下文切换 |
| `--cuda-flush-interval` | 毫秒，默认 `0` | CUDA 数据多久落盘一次（调采集开销/内存的平衡） |
| `--command-file` | 文件 | 从文件读 nsys 开关（命令行可覆盖） |

---

## 4. `nsys stats` —— 把报告读成文字表

```
nsys stats [<args>] <input.nsys-rep | input.sqlite>
```

| 选项 | 取值 / 默认 | 含义 | 例子 |
| --- | --- | --- | --- |
| `-r, --report` | report 名，逗号分隔 | 只看指定的统计表（不指定则打印一组默认表） | `--report cuda_gpu_kern_sum` |
| `-f, --format` | `column`(终端默认) / `table` / `csv` / `tsv` / `json` / `hdoc` | 输出格式 | `--format csv` |
| `-o, --output` | 文件名 / `.`(终端) / `-`(stdout) | 把结果写到文件 | `-o /tmp/stats` |
| `--force-overwrite` | `true`/`false` | 覆盖输出 | |
| `--filter-nvtx` | `range[@domain][/index]` | 只统计某个 NVTX 区间内的事件 | `--filter-nvtx "1D coarsening"` |
| `--filter-time` | `[start]/[end]`（纳秒，可带 `ms`/`us` 单位） | 只统计某段时间 | `--filter-time 1ms/5ms` |
| `-q, --quiet` | — | 少打日志 | |

### 4.1 最该看的 report 名

| report | 内容 | 能看出什么 |
| --- | --- | --- |
| `cuda_gpu_kern_sum` | 各 **kernel** 总耗时/次数/占比 | 哪个 kernel 最热（→ 拿去 ncu） |
| `cuda_gpu_mem_time_sum` | 各类 **memcpy/memset** 耗时 | H2D/D2H 拷贝是不是大头 |
| `cuda_gpu_mem_size_sum` | 拷贝的**数据量** | 传输量是否合理 |
| `cuda_api_sum` | 各 **CUDA API** 耗时 | 是否卡在 `cudaMalloc`/`cudaDeviceSynchronize` |
| `osrt_sum` | OS 运行时调用耗时 | CPU 是否在大量等待 |

> 列出某报告支持的全部 report 名：`nsys stats --help-reports`。

---

## 5. 怎么读输出结果

### 5.1 `cuda_gpu_kern_sum`（kernel 汇总）

典型列：

```text
Time(%)  Total Time(ns)  Instances  Avg(ns)  Med  Min  Max  StdDev  Name
```

- **Time(%) / Total Time**：该 kernel 占 GPU 总时间的比例——**占比最高的优先优化**。
- **Instances**：调用次数。异常多可能是循环里反复启动小 kernel。
- **Avg/Med/Min/Max**：单次耗时分布。
- **Name**：kernel 名（拿去 `ncu -k <Name>`）。

### 5.2 `cuda_gpu_mem_time_sum`（拷贝汇总）

按 `[CUDA memcpy HtoD]` / `[DtoH]` / `[memset]` 分类列耗时。判读：

- 若 H2D/D2H 拷贝耗时和 kernel 计算耗时**同一量级甚至更大** → 瓶颈在**数据传输**，不是计算。
- 对策：减少拷贝、用 pinned 内存（`cudaMallocHost`）、用 stream 让传输与计算重叠（见 `cuda_functions_handbook.md`）。

> 例：lesson_1 向量加法 n=200M，H2D 传两个 800MB 数组。`nsys stats` 一看就知道**拷贝远大于计算**——这是访存密集型算子的典型特征。

### 5.3 时间线（GUI）

`.nsys-rep` 下载到本地用 **Nsight Systems GUI** 打开，看到分行的时间线：CPU 线程、CUDA API、各 stream 的 kernel、memcpy。重点观察：

- kernel 之间有没有**空隙**（GPU 闲置 → CPU 没及时喂下一个）。
- H2D/计算/D2H 是**串行**还是**重叠**（重叠说明用好了 stream）。

---

## 6. 用 NVTX 给时间线打标记（强烈推荐）

裸时间线只有 kernel 名，host 端的逻辑阶段（初始化、对拍、各组实验）看不出来。用 NVTX 手动打区间：

```cpp
#include <nvtx3/nvToolsExt.h>   // CUDA 自带

nvtxRangePush("init data");
/* ... 初始化 ... */
nvtxRangePop();

nvtxRangePush("1D coarsening");
matrixMulKernel<<<grid1d, block>>>(...);
cudaDeviceSynchronize();
nvtxRangePop();
```

编译（通常头文件方式即可，链接报错再加 `-lnvToolsExt`）：

```bash
nvcc -arch=sm_89 -lineinfo 5/lesson_5.cu -o /tmp/lesson_5
```

采集时带上 `nvtx`，之后区间会出现在时间线和 `nsys stats` 的 `nvtx_*` 报告里：

```bash
nsys profile -t cuda,nvtx -o /tmp/l5 -f true /tmp/lesson_5
```

---

## 7. 完整示例（本仓库）

```bash
nvcc -arch=sm_89 -lineinfo 5/lesson_5.cu -o /tmp/lesson_5

# (1) 最常用：采时间线 + 跑完直接打印统计
nsys profile -t cuda -o /tmp/l5 -f true --stats=true /tmp/lesson_5

# (2) 只看 kernel 汇总和拷贝汇总两张表
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum /tmp/l5.nsys-rep

# (3) 导出 CSV 便于记录/对比
nsys stats --report cuda_gpu_kern_sum --format csv -o /tmp/l5_kern /tmp/l5.nsys-rep

# (4) lesson_6 卷积：看 5 个 kernel 的相对耗时
nsys profile -t cuda -o /tmp/l6 -f true /tmp/lesson_6
nsys stats --report cuda_gpu_kern_sum /tmp/l6.nsys-rep

# (5) 只采 cudaProfilerStart/Stop 圈定的区域
nsys profile -t cuda -c cudaProfilerApi -o /tmp/region /tmp/lesson_5
```

---

## 8. 常见问题

| 现象 | 原因 / 解决 |
| --- | --- |
| 报告里 kernel 耗时几乎为 0 | kernel 启动后没同步就退出，或采集范围没覆盖到。确认有 `cudaDeviceSynchronize()` |
| 终端无 GUI | 用 `nsys stats` 看文字表；`.nsys-rep` 拷回本地用 GUI 打开 |
| 想对比两次优化 | 用不同 `-o` 各存一份，分别 `nsys stats`，或 GUI 里并排 |
| 报告太大 | `-d` 限时 / `-c cudaProfilerApi` 圈范围 / `-s none` 关 CPU 采样 |
| 看不到自定义阶段 | 代码加 NVTX（§6），采集时 `-t cuda,nvtx` |
| 权限问题 | nsys 基本不需要特殊权限（不像 ncu）；`system-wide` 类选项才需 root |

---

## 9. 一句话工作流

```text
nvcc -lineinfo 编译
  → nsys profile -t cuda,nvtx --stats=true 采时间线
    → nsys stats 看 cuda_gpu_kern_sum / cuda_gpu_mem_time_sum
      → 拷贝是大头? 减少拷贝 / pinned / stream 重叠
      → 某 kernel 最热? 记下名字 → 交给 ncu（见 ncu_handbook.md）
      → kernel 间有空隙 / 没重叠? 在 GUI 时间线继续查调度
```
