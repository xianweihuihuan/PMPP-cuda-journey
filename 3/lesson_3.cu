#include <cuda_runtime.h>
#include <iostream>

int main(){
  int deviceCount;
  cudaError_t result = cudaGetDeviceCount(&deviceCount);
  if (result != cudaSuccess) {
    printf("cudaGetDeviceCount failed, reason: %s\n", cudaGetErrorString(result));
    return 1;
  }

  printf("device count: %d\n", deviceCount);
  cudaDeviceProp devprop;
  for (int i = 0; i < deviceCount; ++i){
    result = cudaGetDeviceProperties(&devprop, i);
    if (result != cudaSuccess) {
      printf("cudaGetDeviceProperties of device %d failed, reason: %s\n",
             i,
             cudaGetErrorString(result));
      continue;
    }

    printf("\n===== device %d =====\n", i);
    printf("name: %s\n", devprop.name);
    printf("compute capability: %d.%d\n", devprop.major, devprop.minor);
    printf("total global memory: %.2f GB\n",
           devprop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);

    printf("SM count: %d\n", devprop.multiProcessorCount);
    printf("warp size: %d\n", devprop.warpSize);

    printf("shared memory per SM: %zu bytes\n",
           devprop.sharedMemPerMultiprocessor);
    printf("shared memory per block: %zu bytes\n", devprop.sharedMemPerBlock);

    printf("registers per SM: %d\n", devprop.regsPerMultiprocessor);
    printf("registers per block: %d\n", devprop.regsPerBlock);

    printf("max resident blocks per SM: %d\n", devprop.maxBlocksPerMultiProcessor);
    printf("max resident threads per SM: %d\n",
           devprop.maxThreadsPerMultiProcessor);
    printf("max threads per block: %d\n", devprop.maxThreadsPerBlock);
    printf("max resident threads on GPU: %d\n",
           devprop.multiProcessorCount * devprop.maxThreadsPerMultiProcessor);

    printf("max block dimensions: (%d, %d, %d)\n",
           devprop.maxThreadsDim[0],
           devprop.maxThreadsDim[1],
           devprop.maxThreadsDim[2]);
    printf("max grid dimensions: (%d, %d, %d)\n",
           devprop.maxGridSize[0],
           devprop.maxGridSize[1],
           devprop.maxGridSize[2]);
  }

  return 0;
}
