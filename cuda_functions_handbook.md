# CUDA 常用函数手册

> CUDA Runtime API 与设备端内置函数速查。每个函数给出**签名 / 参数 / 作用 / 示例 / 注意**，方便和 `1/`~`6/` 各课对照。
> 约定：绝大多数 runtime 函数返回 `cudaError_t`，成功时为 `cudaSuccess`，**每次调用都应检查返回值**（见 [§9 错误处理](#9-错误处理) 与 [§12 错误检查封装](#12-统一的错误检查封装)）。

## 目录

1. [函数与变量限定符](#1-函数与变量限定符)
2. [设备查询与管理](#2-设备查询与管理)
3. [显存分配与释放](#3-显存分配与释放)
4. [主机内存（pinned / 统一内存）](#4-主机内存pinned--统一内存)
5. [数据传输](#5-数据传输)
6. [常量内存](#6-常量内存)
7. [核函数启动与执行配置](#7-核函数启动与执行配置)
8. [同步](#8-同步)
9. [错误处理](#9-错误处理)
10. [流（Stream）](#10-流stream)
11. [事件与计时](#11-事件与计时)
12. [设备端内置变量与函数](#12-设备端内置变量与函数)
13. [统一的错误检查封装](#13-统一的错误检查封装)
14. [一个完整的最小流程](#14-一个完整的最小流程)

---

## 1. 函数与变量限定符

### 函数限定符

| 限定符 | 执行位置 | 调用位置 | 说明 |
| --- | --- | --- | --- |
| `__global__` | Device | Host | kernel 函数，用 `<<<grid, block>>>` 启动，返回值必须为 `void` |
| `__device__` | Device | Device | 只能被 GPU 代码调用，常用于拆分 kernel 内部逻辑 |
| `__host__` | Host | Host | 普通 CPU 函数，默认即是，可省略 |

`__host__ __device__` 可组合，让同一函数同时生成 CPU 和 GPU 两个版本：

```cpp
__host__ __device__ float add(float a, float b) { return a + b; }
```

### 变量限定符

| 限定符 | 所在内存 | 作用域 | 生命周期 |
| --- | --- | --- | --- |
| （无，普通局部变量） | register | thread | kernel |
| `__shared__` | shared memory | block | kernel |
| `__constant__` | constant memory | grid（只读） | application |
| `__device__` | global memory | grid | application |

---

## 2. 设备查询与管理

### cudaGetDeviceCount

```cpp
cudaError_t cudaGetDeviceCount(int* count)
```

- `count`：输出型参数，返回可见 GPU 数量。
- `return`：错误码。

作用：查询系统中可用的 CUDA 设备数量。多卡程序据此遍历每张卡。

```cpp
int deviceCount;
cudaGetDeviceCount(&deviceCount);
printf("device count: %d\n", deviceCount);
```

### cudaGetDeviceProperties

```cpp
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
```

- `prop`：输出型参数，填入设备属性结构体。
- `device`：设备编号（`0` ~ `count-1`）。
- `return`：错误码。

作用：查询某张 GPU 的硬件属性（第四章重点）。

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("name: %s\n", prop.name);
printf("compute capability: %d.%d\n", prop.major, prop.minor);
printf("SM count: %d\n", prop.multiProcessorCount);
```

`cudaDeviceProp` 常用字段：

| 字段 | 含义 |
| --- | --- |
| `name` | 设备名称 |
| `major` / `minor` | 计算能力 |
| `multiProcessorCount` | SM 数量 |
| `warpSize` | warp 线程数（通常 32） |
| `maxThreadsPerBlock` | 单 block 最大线程数（通常 1024） |
| `maxThreadsPerMultiProcessor` | 单 SM 最大驻留线程数 |
| `maxBlocksPerMultiProcessor` | 单 SM 最大驻留 block 数 |
| `sharedMemPerBlock` | 单 block 可用 shared memory |
| `sharedMemPerMultiprocessor` | 单 SM 的 shared memory 总量 |
| `regsPerBlock` / `regsPerMultiprocessor` | 寄存器上限 |
| `totalGlobalMem` | 全局内存总量（字节） |

### cudaSetDevice / cudaGetDevice

```cpp
cudaError_t cudaSetDevice(int device)
cudaError_t cudaGetDevice(int* device)
```

作用：`cudaSetDevice` 设置当前 host 线程后续操作使用哪张 GPU；`cudaGetDevice` 查询当前使用的设备编号。多卡时每张卡操作前要先 `cudaSetDevice`。

```cpp
cudaSetDevice(1);     // 之后的 cudaMalloc / kernel 都落在 1 号卡
```

### cudaDeviceReset

```cpp
cudaError_t cudaDeviceReset(void)
```

作用：销毁并清理当前设备上的所有资源（相当于「重置 GPU」）。常放在程序末尾，确保 profiler 能拿到完整数据、显存被干净释放。

```cpp
cudaDeviceReset();
```

---

## 3. 显存分配与释放

### cudaMalloc

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size)
```

- `devPtr`：输出型参数（二级指针），返回设备内存地址。
- `size`：申请的**字节数**。
- `return`：错误码。

作用：在 GPU global memory 上申请 `size` 字节。

```cpp
float* A_d;
int size = sizeof(float) * n;
cudaMalloc((void**)&A_d, size);
```

注意：

- 第一个参数是 `void**`，所以写成 `(void**)&A_d`——它要把申请到的地址**写回**给调用者。
- `size` 是字节数，记得乘 `sizeof(...)`。
- 每个 `cudaMalloc` 都要有对应的 `cudaFree`，否则显存泄漏。

### cudaFree

```cpp
cudaError_t cudaFree(void* devPtr)
```

- `devPtr`：之前 `cudaMalloc` 得到的设备指针。
- `return`：错误码。

作用：释放设备内存。

```cpp
cudaFree(A_d);
```

### cudaMemset

```cpp
cudaError_t cudaMemset(void* devPtr, int value, size_t count)
```

- `devPtr`：设备内存地址。
- `value`：填充值（**按字节**填充，取 `value` 的低 8 位）。
- `count`：填充字节数。
- `return`：错误码。

作用：把一段显存按字节填成某个值，最常用来清零。

```cpp
cudaMemset(C_d, 0, size);   // 把 C_d 清零
```

注意：因为是**按字节**填充，`cudaMemset(p, 1, n*sizeof(float))` 不会把 float 数组填成 `1.0f`。清零（填 0）才是安全的常见用法。

---

## 4. 主机内存（pinned / 统一内存）

### cudaMallocHost / cudaFreeHost

```cpp
cudaError_t cudaMallocHost(void** ptr, size_t size)
cudaError_t cudaFreeHost(void* ptr)
```

作用：申请 / 释放 **pinned（页锁定）主机内存**。pinned 内存不会被操作系统换页，`cudaMemcpy` 传输更快，且是 `cudaMemcpyAsync` 真正异步的前提。

```cpp
float* h_pinned;
cudaMallocHost((void**)&h_pinned, size);   // 比 malloc 的可分页内存传输更快
...
cudaFreeHost(h_pinned);
```

注意：pinned 内存占用物理内存、申请较慢，不要滥用；普通数据用 `malloc` / `new` 即可。

### cudaMallocManaged

```cpp
cudaError_t cudaMallocManaged(void** devPtr, size_t size,
                              unsigned int flags = cudaMemAttachGlobal)
```

作用：申请 **统一内存（Unified Memory）**，返回的指针 CPU 和 GPU 都能直接访问，数据迁移由驱动自动完成，省去手动 `cudaMemcpy`。

```cpp
float* data;
cudaMallocManaged((void**)&data, size);
data[0] = 1.0f;                 // CPU 直接写
kernel<<<grid, block>>>(data);  // GPU 直接用
cudaDeviceSynchronize();        // 用完要同步再回 CPU 读
cudaFree(data);                 // 仍用 cudaFree 释放
```

注意：用法简单，但自动迁移可能带来隐藏的拷贝开销；性能关键路径仍建议显式 `cudaMalloc` + `cudaMemcpy`。

---

## 5. 数据传输

### cudaMemcpy

```cpp
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
```

- `dst`：目标地址。
- `src`：源地址。
- `count`：拷贝字节数。
- `kind`：拷贝方向。
- `return`：错误码。

作用：在 host 和 device 之间（或各自内部）拷贝数据。

`kind` 取值：

| `kind` | 方向 |
| --- | --- |
| `cudaMemcpyHostToDevice` | 主机 → 设备（上传输入） |
| `cudaMemcpyDeviceToHost` | 设备 → 主机（取回结果） |
| `cudaMemcpyDeviceToDevice` | 设备内拷贝 |
| `cudaMemcpyHostToHost` | 主机内拷贝 |

```cpp
cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);   // 上传
kernel<<<grid, block>>>(A_d, ...);
cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);   // 取回
```

注意：`cudaMemcpy` 是**同步**的——它会等之前的 kernel 跑完、并等拷贝完成才返回。所以取回结果前通常不需要再额外同步。

### cudaMemcpyAsync

```cpp
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream = 0)
```

作用：异步拷贝，发起后立即返回，拷贝在指定 `stream` 中排队执行，可与计算重叠。

```cpp
cudaMemcpyAsync(A_d, A_h, size, cudaMemcpyHostToDevice, stream);
```

注意：真正异步要求**主机端是 pinned 内存**（见 `cudaMallocHost`），否则会退化为同步。

---

## 6. 常量内存

### cudaMemcpyToSymbol / cudaMemcpyFromSymbol

```cpp
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count,
                               size_t offset = 0,
                               cudaMemcpyKind kind = cudaMemcpyHostToDevice)
cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count,
                                 size_t offset = 0,
                                 cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
```

- `symbol`：设备端**符号名**（`__constant__` 或 `__device__` 全局变量），直接写变量名。
- `src` / `dst`：host 端源 / 目标地址。
- `count`：字节数。
- `return`：错误码。

作用：在 host 和「具名设备符号」之间拷贝，最常用于给常量内存赋值（第七章卷积的 filter）。

```cpp
__constant__ float F[5][5];                 // 文件作用域声明
...
cudaMemcpyToSymbol(F, h_F, 5 * 5 * sizeof(float));   // 第一个参数直接写符号名
```

注意：常量内存适合**小、只读、所有线程访问同一批元素**的数据（如卷积 filter）。详见 `6/lesson_6.md` §7.3。

---

## 7. 核函数启动与执行配置

### `<<< >>>` 启动语法

```cpp
kernel<<<gridDim, blockDim, sharedBytes, stream>>>(args...);
```

| 参数 | 含义 | 是否必填 |
| --- | --- | --- |
| `gridDim` | grid 中 block 数量（`dim3` 或 int） | 必填 |
| `blockDim` | 每个 block 的线程数（`dim3` 或 int） | 必填 |
| `sharedBytes` | 动态 shared memory 字节数 | 可选，默认 0 |
| `stream` | 所属 stream | 可选，默认 0 |

```cpp
// 一维：向量加法（lesson_1）
AddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

// 二维：矩阵乘法（lesson_4）
dim3 block(16, 16, 1);
dim3 grid((width + 15) / 16, (width + 15) / 16, 1);
matrixMulKernel<<<grid, block>>>(M_d, N_d, P_d, width);

// 带动态 shared memory（lesson_4）
kernel<<<grid, block, sharedBytes>>>(...);   // 配合 extern __shared__
```

注意：

- `(width + TILE - 1) / TILE` 是**向上取整**的 block 数，保证覆盖所有数据；配合 kernel 里的 `if (i < n)` 防越界。
- kernel 启动是**异步**的：host 发出后立即返回，不等 GPU 执行完。
- 启动本身不返回 `cudaError_t`，要用 `cudaGetLastError()` 抓启动错误。

### dim3

```cpp
struct dim3 { unsigned x, y, z; };   // 未指定的维度默认为 1
```

作用：描述 grid / block 的三维形状。

```cpp
dim3 block(32, 8);     // x=32, y=8, z=1，共 256 线程
```

---

## 8. 同步

### __syncthreads（设备端）

```cpp
void __syncthreads()
```

作用：**block 内屏障**——block 内所有线程都到达此处后才继续。用于保护 shared memory 的读写顺序（tiling 的两个屏障，见 `4/lesson_4.md` §5.4）。

```cpp
N_s[ty][tx] = N[...];   // 协作加载
__syncthreads();        // 等所有人加载完，再开始算
... 计算 ...
__syncthreads();        // 等所有人算完，再覆盖 shared memory
```

注意：

- 只同步**同一个 block**，不能跨 block。
- block 内**所有线程都必须执行到同一句** `__syncthreads()`，否则死锁——不要放进只有部分线程进入的 `if` 分支。

### __syncwarp（设备端）

```cpp
void __syncwarp(unsigned mask = 0xffffffff)
```

作用：**warp 内同步**，比 `__syncthreads()` 粒度更细。在做 warp 级优化（如 warp 内 reduction）时使用。

### cudaDeviceSynchronize（host 端）

```cpp
cudaError_t cudaDeviceSynchronize(void)
```

作用：阻塞 host，直到设备上所有已提交任务完成。常用于：等 kernel 真正跑完、计时、或捕获 kernel 运行期错误。

```cpp
kernel<<<grid, block>>>(...);
cudaDeviceSynchronize();   // 等 kernel 执行结束
```

### cudaStreamSynchronize（host 端）

```cpp
cudaError_t cudaStreamSynchronize(cudaStream_t stream)
```

作用：只阻塞 host 直到指定 `stream` 的任务完成，比 `cudaDeviceSynchronize` 范围更小。

---

## 9. 错误处理

### cudaGetLastError / cudaPeekAtLastError

```cpp
cudaError_t cudaGetLastError(void)       // 返回并「清除」最近错误
cudaError_t cudaPeekAtLastError(void)    // 返回但「不清除」最近错误
```

作用：取最近一次 CUDA 错误。因为 kernel 启动不返回错误码，要靠它来检查。

```cpp
kernel<<<grid, block>>>(...);
cudaGetLastError();        // 抓「启动配置」错误（如 block 超过 1024）
cudaDeviceSynchronize();   // 抓「执行期」错误（如越界访问）
```

### cudaGetErrorString / cudaGetErrorName

```cpp
const char* cudaGetErrorString(cudaError_t error)   // 人类可读的描述
const char* cudaGetErrorName(cudaError_t error)     // 错误码枚举名
```

作用：把错误码转成字符串，便于打印。

```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess)
  printf("CUDA error: %s\n", cudaGetErrorString(err));
```

---

## 10. 流（Stream）

流是一串按顺序执行的 GPU 操作；不同流之间可以并发，从而让传输与计算重叠。

### cudaStreamCreate / cudaStreamDestroy

```cpp
cudaError_t cudaStreamCreate(cudaStream_t* pStream)
cudaError_t cudaStreamDestroy(cudaStream_t stream)
```

作用：创建 / 销毁一个流。

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(d, h, size, cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d);     // 在该流里执行
cudaStreamSynchronize(stream);

cudaStreamDestroy(stream);
```

注意：不指定 stream 时用的是默认流（`0`）。多流 + pinned 内存 + `cudaMemcpyAsync` 才能实现传输/计算重叠。

---

## 11. 事件与计时

测 kernel 耗时的标准做法（比 host 端 `clock()` 更准，因为 kernel 异步）。

```cpp
cudaError_t cudaEventCreate(cudaEvent_t* event)
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0)
cudaError_t cudaEventSynchronize(cudaEvent_t event)
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
cudaError_t cudaEventDestroy(cudaEvent_t event)
```

| 函数 | 作用 |
| --- | --- |
| `cudaEventCreate` | 创建事件 |
| `cudaEventRecord` | 在流中「打一个时间戳」 |
| `cudaEventSynchronize` | 等事件真正发生 |
| `cudaEventElapsedTime` | 计算两事件间毫秒数 |
| `cudaEventDestroy` | 销毁事件 |

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);                 // 必须等 stop 发生
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);     // 得到耗时（毫秒）
printf("kernel time: %f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 12. 设备端内置变量与函数

只能在 `__global__` / `__device__` 函数里使用。

### 内置索引变量（`dim3` 类型，含 `.x / .y / .z`）

| 变量 | 含义 |
| --- | --- |
| `threadIdx` | 线程在 block 内的索引 |
| `blockIdx` | block 在 grid 内的索引 |
| `blockDim` | 每个 block 的线程数 |
| `gridDim` | grid 的 block 数 |

全局索引标准写法：

```cpp
int i   = blockIdx.x * blockDim.x + threadIdx.x;   // 一维
int Row = blockIdx.y * blockDim.y + threadIdx.y;   // 二维行
int Col = blockIdx.x * blockDim.x + threadIdx.x;   // 二维列
```

### 原子操作

多个线程写同一地址时必须用原子操作，避免竞态。返回值是**修改前**的旧值。

```cpp
T atomicAdd(T* address, T val)    // *address += val，返回旧值
```

| 函数 | 作用 |
| --- | --- |
| `atomicAdd` | 原子加 |
| `atomicSub` | 原子减 |
| `atomicMax` / `atomicMin` | 原子取最大 / 最小 |
| `atomicExch` | 原子交换 |
| `atomicCAS(addr, compare, val)` | 比较并交换（实现自定义原子操作的基石） |

```cpp
atomicAdd(&histogram[bin], 1);   // 直方图：多线程往同一 bin 累加
```

### 常用数学函数

设备端单精度版带 `f` 后缀：`sqrtf` / `expf` / `logf` / `sinf` / `cosf` / `powf` / `fabsf` / `fminf` / `fmaxf`。

带 `__` 前缀的是更快、精度略低的 intrinsic：`__expf` / `__logf` / `__sinf` / `__fdividef` 等。

---

## 13. 统一的错误检查封装

`lesson_4.cu` 的封装写法，建议每个项目都包一层：

```cpp
bool checkCuda(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    std::cout << message << " failed, reason: "
              << cudaGetErrorString(result) << "\n";
    return false;
  }
  return true;
}

checkCuda(cudaMalloc((void**)&M_d, size), "cuda malloc of M_d");
checkCuda(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice), "copy M_h to M_d");
```

也可写成自动带文件名 / 行号的宏：

```cpp
#define CUDA_CHECK(call)                                              \
  do {                                                               \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
              cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                            \
    }                                                                \
  } while (0)

CUDA_CHECK(cudaMalloc((void**)&d_ptr, size));
```

---

## 14. 一个完整的最小流程

```cpp
// 1. 分配显存
cudaMalloc((void**)&d_in,  size);
cudaMalloc((void**)&d_out, size);

// 2. 上传输入
cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

// 3. 启动 kernel
dim3 block(256);
dim3 grid((n + block.x - 1) / block.x);
myKernel<<<grid, block>>>(d_in, d_out, n);
cudaGetLastError();           // 检查启动配置
cudaDeviceSynchronize();      // 等待执行 + 检查运行期错误

// 4. 取回结果
cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

// 5. 释放
cudaFree(d_in);
cudaFree(d_out);
cudaDeviceReset();
```
