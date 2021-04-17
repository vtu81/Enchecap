#ifndef MATRIXMULCUBLAS_H__
#define MATRIXMULCUBLAS_H__

#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
// Enchecap
#include "Enchecap_device.h"

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

int beginMatrixMulCUBLAS(int argc, char **argv, ECPreg ecpreg);

#endif