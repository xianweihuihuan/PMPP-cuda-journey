# Nsight Compute (`ncu`) 命令行参考手册

> 环境：Nsight Compute **2026.1** / RTX 4060 Laptop (sm_89) / CUDA `/usr/local/cuda`。
> 本文档逐选项解释 `ncu --help` 的含义、取值、默认值、用法和输出结果含义，配例子。
> 命令示例统一用本仓库的程序（`/tmp/lesson_5`、`/tmp/lesson_6`）。

---

## 1. 这个工具是干什么的

`ncu` 是 **单个 CUDA kernel 的性能显微镜**。它通过读取 GPU 硬件性能计数器，给出一个 kernel 内部的详细指标：计算/访存吞吐、occupancy、warp 停顿原因、各级缓存命中率、寄存器与 shared memory 用量，并能把瓶颈对应回源码行。

**工作原理（必须理解）**：一次运行采不全所有计数器，所以 ncu 会把**同一个 kernel 重放（replay）多遍**，每遍采一部分指标。因此：

- 开销很大（比直接运行慢几十倍）。
- **必须用 `-k` / `-c` 限定只测某个 kernel 的某几次启动**，否则会把整个程序的每个 kernel 都重放，极慢。

```
ncu [options] program [program-args]
```

---

## 2. General Options（通用）

| 选项 | 取值 / 默认 | 含义 | 例子 |
| --- | --- | --- | --- |
| `-h, --help` | — | 打印帮助 | `ncu --help` |
| `-v, --version` | — | 打印版本号 | `ncu --version` |
| `--mode` | `launch-and-attach`(默认) / `launch` / `attach` | 与目标程序的交互方式。默认就是「启动并附加分析」；`launch` 启动后挂起等待，`attach` 附加到已启动进程（用于远程/GUI 场景） | `ncu --mode=launch myApp` |
| `-p, --port` | 默认 `49152` | 连接目标程序的基础端口（远程 attach 用） | |
| `--config-file` | 默认开 | 是否读取 `config.ncu-cfg` 配置文件 | |

> MPS / Multi-process Communicator / Metric Distributor 等选项是给 MPS 与多进程协同场景的，单机单进程学习用不到，本手册略。

---

## 3. Launch Options（启动控制）

| 选项 | 取值 / 默认 | 含义 | 例子 |
| --- | --- | --- | --- |
| `--check-exit-code` | 默认 `1`（开） | 检查程序退出码，非 0 时报错；application replay 模式下首遍非 0 即停 | |
| `--nvtx` | — | 启用 NVTX 支持（配合 `--nvtx-include/-exclude` 按代码里的 NVTX 区间筛 kernel） | `ncu --nvtx --nvtx-include "1D coarsening" ...` |
| `--call-stack` | — | 采集 CPU 调用栈（知道 kernel 是从哪段 host 代码发起的） | |
| `--call-stack-type` | `native`(默认) / `python` | 调用栈类型，可多选；隐含 `--call-stack` | |
| `--target-processes` | `all`(默认) / `application-only` | 是否连子进程一起分析。`application-only` 只分析主进程 | `--target-processes application-only` |
| `--target-processes-filter` | `<名字>` / `regex:<…>` / `exclude:<…>` | 按进程名筛选要分析的进程 | |
| `--null-stdin` | — | 用 `/dev/null` 作标准输入，避免后台运行时因读 stdin 挂起 | |

---

## 4. Common Profile Options（采集核心）

### 4.1 replay 与采集方式

| 选项 | 取值 / 默认 | 含义 |
| --- | --- | --- |
| `--replay-mode` | `kernel`(默认) / `application` / `range` / `app-range` | 重放方式。`kernel`=透明地把单个 kernel 重放多次（最常用）；`application`=整个程序重跑多次（要求程序确定性可复现）；`range`/`app-range`=按范围重放 |
| `--graph-profiling` | `node`(默认) / `graph` | CUDA Graph 的分析粒度：按单个 kernel 节点，还是整张图 |
| `--cache-control` | `all`(默认) / `none` | 每遍重放前是否刷新 GPU 缓存。`all`=刷新，保证每遍从冷缓存开始、结果可复现；`none`=不刷新（想看缓存被预热后的真实表现时用） |
| `--clock-control` | `boost`(默认) / `base` / `force-boost` / `none` / `reset` | 分析期间锁定 GPU 时钟。默认锁到 boost 频率，保证 run-to-run 可比；`none`=不锁（看真实动态频率） |

> 默认 `kernel` 重放 + 锁时钟 + 刷缓存 = 给你**可复现、可对比**的测量，这正是优化迭代想要的。

### 4.2 section / set / 指标 选择

| 选项 | 取值 / 默认 | 含义 | 例子 |
| --- | --- | --- | --- |
| `--set` | 不指定时用 `basic` | 采集「指标集」。集越全越慢。常用：`basic`(快) / `default` / `full`(全，含源码级) | `--set full` |
| `--list-sets` | — | 列出所有可用的 set | `ncu --list-sets` |
| `--section` | `<名字>` / `regex:<…>` | 只采指定的 section（比整组 set 更省时间）。可多次给 | `--section SpeedOfLight --section Occupancy` |
| `--list-sections` | — | 列出所有 section | `ncu --list-sections` |
| `--metrics` | 逗号分隔；支持 `regex:` / `group:` / `breakdown:` 前缀 | 精确指定要采的底层 metric | `--metrics sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| `--list-metrics` | — | 列出当前所选 section 会采的 metric | |
| `--query-metrics` | — | 查询本机设备支持的所有 metric | |

常用 section 一览：

| section | 回答的问题 |
| --- | --- |
| `SpeedOfLight` | compute-bound 还是 memory-bound？离峰值多远？ |
| `MemoryWorkloadAnalysis` | 访存是否合并？L1/L2/DRAM 各承担多少？ |
| `Occupancy` | 实际 occupancy？被寄存器/shared/block 哪个限制？ |
| `LaunchStats` | grid/block、每线程寄存器、每 block shared memory |
| `SchedulerStats` / `WarpStateStats` | warp 为什么 stall（等访存？等 `__syncthreads`？） |
| `ComputeWorkloadAnalysis` | 各计算管线（FP32/FMA…）利用率 |
| `SourceCounters` | 把指标摊到源码行/SASS（需 `-lineinfo`） |

### 4.3 规则与源码

| 选项 | 取值 / 默认 | 含义 |
| --- | --- | --- |
| `--apply-rules` | 默认 `1`（开） | 是否对采集结果套用「分析规则」，生成 OPT/WRN 等优化建议 |
| `--rule` | `<规则名>` | 只应用某条规则（隐含开 apply-rules） |
| `--import-sass` | 默认 `1`（开） | 把 SASS/PTX/cubin 元信息存进报告 |
| `--import-source` | 默认 `0`（关） | 把 `-lineinfo` 关联的 CUDA 源码**永久**嵌入报告（方便换机也能看源码热点） |
| `--source-folders` | 逗号分隔路径 | import-source 时去哪找源码 |

### 4.4 采集起止

| 选项 | 取值 / 默认 | 含义 |
| --- | --- | --- |
| `--profile-from-start` | 默认 `1`（开） | 是否从程序一开始就分析。设 `off` 时配合代码里的 `cudaProfilerStart/Stop` 圈定区域 |
| `--disable-profiler-start-stop` | — | 忽略代码里的 `cu(da)ProfilerStart/Stop` |
| `--quiet` | — | 静默，抑制所有输出 |
| `--verbose` | — | 更啰嗦的输出 |

---

## 5. Filter Profile Options（筛选要测哪个 kernel）★最重要

| 选项 | 取值 / 默认 | 含义 | 例子 |
| --- | --- | --- | --- |
| `-k, --kernel-name` | `<名字>` 精确 / `regex:<…>` | **按名字筛 kernel**。学习时几乎必加 | `-k matrixMulKernel` / `-k "regex:matrixMul"` |
| `--kernel-id` | `ctx:stream:[op:]name:invocation` | 更精细的匹配（指定 context/stream/第几次调用） | `--kernel-id ::matrixMulKernel:2`（第 2 次调用） |
| `--kernel-name-base` | `function`(默认) / `demangled` / `mangled` | `-k` 匹配的名字基准 | |
| `-c, --launch-count` | 整数 | **只采集匹配 kernel 的前 N 次启动**。配 `-k` 用 `-c 1` 表示每个匹配 kernel 只测一次 | `-c 1` |
| `-s, --launch-skip` | 默认 `0` | 跳过前 N 次匹配的启动再开始测（计数只算匹配的） | `-s 2` |
| `--launch-skip-before-match` | 默认 `0` | 跳过前 N 次启动（计数算**所有**启动，不只匹配的） | |
| `--devices` | 逗号分隔 | 只在指定 GPU 上分析 | `--devices 0` |
| `--filter-mode` | `global`(默认) / `per-gpu` / `per-launch-config` | 筛选作用范围 | |
| `--nvtx-include` / `--nvtx-exclude` | NVTX 区间名 | 按代码里的 NVTX 区间筛 kernel（需 `--nvtx`） | `--nvtx --nvtx-include "1D coarsening"` |
| `--range-filter` | `<yes/no>:<start/stop实例>:<nvtx实例>` | 只测匹配区间的第几个实例 | |

> 记住这条铁律：**`-k <kernel> -c 1`**。它把「整个程序逐 kernel 重放」压成「只测这一个 kernel 一次」，分析时间从分钟级降到秒级。

---

## 6. File Options（输出/导入文件）

| 选项 | 取值 / 默认 | 含义 | 例子 |
| --- | --- | --- | --- |
| `-o, --export` | 文件名 | 把结果存成 `.ncu-rep` 报告（不设则用临时文件、用完即删） | `-o /tmp/mm` → 生成 `/tmp/mm.ncu-rep` |
| `-f, --force-overwrite` | — | 覆盖已存在的输出文件 | `-f` |
| `-i, --import` | 文件名 | 读取已有报告（事后在命令行重新看） | `ncu -i /tmp/mm.ncu-rep --page details` |
| `--log-file` | 文件名 / `stdout` / `stderr` | 把工具文字输出重定向到文件 | `--log-file /tmp/ncu.log` |
| `--open-in-ui` | — | 直接在 GUI 里打开结果，而不是打印到终端 | |

---

## 7. Console Output Options（终端输出怎么显示）

| 选项 | 取值 / 默认 | 含义 | 例子 |
| --- | --- | --- | --- |
| `--page` | `details`(默认) / `raw` / `source` / `session` | 看哪一页。`details`=分 section 的指标+规则；`raw`=所有原始 metric；`source`=源码级；`session`=设备/会话属性 | `--page raw` |
| `--csv` | — | 用 CSV 输出（便于后处理/导表） | `--csv` |
| `--print-source` | `sass` / `ptx` / `cuda` / `cuda,sass` | source 页显示哪种源码视图 | `--page source --print-source sass` |
| `--print-details` | `header`(默认) / `body` / `all` | details 页显示 section 的哪部分指标 | `--print-details all` |
| `--print-summary` | `none`(默认) / `per-gpu` / `per-kernel` / `per-nvtx` | 汇总模式。`per-kernel`=把同名 kernel 多次启动汇总 | `--print-summary per-kernel` |
| `--print-units` | `auto`(默认) / `base` | 单位是否自动缩放（KB/MB…）还是用基础单位 | |
| `--print-metric-name` | `label`(默认) / `name` / `label-name` | 指标列显示「人类标签」还是「metric 名」 | |
| `--print-fp` | — | 所有数值用浮点显示 | |

---

## 8. 怎么读输出结果

### 8.1 details 页的结构

每个 kernel 一段，按 section 排列。每个 section 是一张 **指标表**（三列：指标标签 / 数值 / 单位），后面常跟一条 **规则消息**，前缀标明级别：

| 前缀 | 含义 |
| --- | --- |
| `OPT` | 优化建议（最该看） |
| `WRN` | 警告 |
| `INF` | 信息提示 |

### 8.2 第一眼：`SpeedOfLight`（定瓶颈）

```text
Compute (SM) Throughput   [%]   ← 计算单元达到了峰值的百分之多少
Memory Throughput         [%]   ← 访存带宽达到了峰值的百分之多少
```

判读（对应 PMPP 第六章方法论）：

- **Memory 高、Compute 低** → memory-bound → 上 tiling / 合并访问 / 线程粗化（PMPP 5、6 章手段）。
- **Compute 高、Memory 低** → compute-bound → 减少指令 / 用 FMA / 降精度。
- **两个都低** → occupancy 不足或延迟没藏住 → 看 §8.4。

### 8.3 `MemoryWorkloadAnalysis`（访存细节）

- **L1/L2 Hit Rate**：缓存命中率。
- **sectors/request**：每次访存请求触及多少 32B 扇区。合并良好时低；偏高=访存不合并（第六章 coalescing）。
- **DRAM Throughput**：到显存的真实带宽。

### 8.4 `Occupancy`（并行度）

- **Achieved Occupancy** vs **Theoretical Occupancy**：实际 / 理论上限。
- 看是被 **Registers Per Thread**、**Shared Memory Per Block** 还是 **Block Limit** 卡住——正是第四、五章的 occupancy 限制因素。

### 8.5 `LaunchStats`

直接列出 grid/block 维度、每线程寄存器数、每 block 动态/静态 shared memory——核对启动配置是否如预期。

---

## 9. 完整示例（本仓库）

```bash
# 必须 -lineinfo 才能把瓶颈对回源码（绝不要用 -G，那是 debug、会关优化）
nvcc -arch=sm_89 -lineinfo 5/lesson_5.cu -o /tmp/lesson_5

# (1) 最常用：只测某 kernel 一次，basic 集，终端看
ncu --set basic -k matrixMulKernel -c 1 /tmp/lesson_5

# (2) 只看定瓶颈 + occupancy 两个 section（更快）
ncu --section SpeedOfLight --section Occupancy -k "regex:matrixMul" -c 1 /tmp/lesson_5

# (3) 全量 + 存报告，GUI 里看源码级热点
ncu --set full -k matrixMulKernel -c 1 -o /tmp/mm -f /tmp/lesson_5
ncu -i /tmp/mm.ncu-rep --page details          # 事后在终端重读
ncu -i /tmp/mm.ncu-rep --page source --print-source cuda,sass

# (4) 对比 lesson_6 的 global filter vs 常量内存版
ncu --set basic -k convolution_2D_Kernel            -c 1 /tmp/lesson_6
ncu --set basic -k convolution_2D_Kernel_by_const   -c 1 /tmp/lesson_6

# (5) CSV 输出便于记录
ncu --set basic -k matrixMulKernel -c 1 --csv /tmp/lesson_5 > /tmp/mm.csv
```

**优化迭代法**：`ncu --set full -k K -c 1 -o v1` 存基线 → 改代码 → 重新对拍确认仍正确 → `... -o v2` → 在 GUI 用 **Add Baseline** 把 v1 设基线，v2 的每个指标显示相对增减。

---

## 10. 常见问题

| 现象 | 原因 / 解决 |
| --- | --- |
| `ERR_NVGPUCTRPERM` 无权限读计数器 | 临时 `sudo ncu …`；长期设驱动 `NVreg_RestrictProfilingToAdminUsers=0` 后重启 |
| 极慢 | 没限范围。加 `-k <kernel> -c 1`，用 `--set basic` 而非 `full` |
| 瓶颈对不到源码行 | 编译漏了 `-lineinfo` |
| 指标全 0 / 采集失败 | kernel 没真正执行（配置错/越界提前退出）；先确保程序本身对拍通过 |
| `-G` vs `-lineinfo` | 性能分析永远用 `-lineinfo`；`-G` 关优化，结果不代表真实性能 |

---

## 11. 一句话工作流

```text
nsys 定位到最耗时 kernel（见 nsys_handbook.md）
  → nvcc -lineinfo 编译
    → ncu --set basic -k <kernel> -c 1   看 SpeedOfLight 定瓶颈
      → memory-bound? 看 MemoryWorkloadAnalysis
      → occupancy 低? 看 Occupancy（寄存器/shared/block 限制）
      → 要源码热点? --set full + --page source
        → 改代码 → 对拍 → -o 存新报告 → GUI 设 baseline 对比
```
