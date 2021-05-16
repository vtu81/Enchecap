#ifndef RSA_H__
#define RSA_H__

#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <time.h>
#include <omp.h>
#define BUZZ_SIZE 10002

void encrypt_gpu(void *ptr, int size, void *gpu_user_keys,unsigned *auxilaryBuf=NULL);
void decrypt_gpu(void *ptr, int size, void *gpu_user_keys,unsigned *auxilaryBuf=NULL);

void encrypt_gpu_old(void *ptr, int size);
void decrypt_gpu_old(void *ptr, int size);

#endif