#include "DeviceApp.h"

__global__ void GPUFunction()
{
  printf("This function is defined to run on the GPU.(threadId:%d, blockId:%d)\n", threadIdx.x, blockIdx.x);
}

void callGPU()
{
  GPUFunction<<<3, 2>>>();
  cudaDeviceSynchronize();
}