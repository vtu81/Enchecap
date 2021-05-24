#include "matrixMulCUBLAS.h"

unsigned long int p, q, n, e, d; /* FIXME */

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

int nIter=1,encrypt=1,verify=1;

void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB);
void randomInit(float *data, int size);
void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol);
void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size);
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size, ECPreg ecpreg);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    devID = findCudaDevice(argc, (const char **)argv);

    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "iter"))
    {
        nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iter");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "encrypt"))
    {
        encrypt = getCmdLineArgumentInt(argc, (const char **)argv, "encrypt");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "verify"))
    {
        verify = getCmdLineArgumentInt(argc, (const char **)argv, "verify");
    }

    iSizeMultiple = min(iSizeMultiple, 100);
    iSizeMultiple = max(iSizeMultiple, 1);

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    int block_size = 32;

    matrix_size.uiWA = 3 * block_size * iSizeMultiple;
    matrix_size.uiHA = 4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 2 * block_size * iSizeMultiple;
    matrix_size.uiHB = 3 * block_size * iSizeMultiple;
    matrix_size.uiWC = 2 * block_size * iSizeMultiple;
    matrix_size.uiHC = 4 * block_size * iSizeMultiple;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size, ECPreg ecpreg)
{
	// p = 157;
	// q = 373;
    // p= 126611;
    // q= 130643;
    // e=0x10001;
    // d=5621128193;
    
    /** FIXME: The keys should be acquired in the following steps:
     * 1. 
     **/
    p = 74531;
    q = 37019;
    e = 0x10001;
	d = 985968293;
	n = p * q;
    struct timespec begin,end;
    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    int block_size = 32;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    float *h_A2 = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);
    float *h_B2 = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A2, size_A);
    randomInit(h_B2, size_B);
    
    memcpy(h_A,h_A2,mem_size_A);
    memcpy(h_B,h_B2,mem_size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_C      = (float *) malloc(mem_size_C);
    float *h_CUBLAS = (float *) malloc(mem_size_C);

    checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    


    // if(encrypt){
    //     encrypt_cpu(h_A,mem_size_A/sizeof(int));
    //     encrypt_cpu(h_B,mem_size_B/sizeof(int));
    // }
    // checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
    // if(encrypt){
    //     decrypt_gpu(d_A,mem_size_A/sizeof(int));
    //     decrypt_gpu(d_B,mem_size_B/sizeof(int));
    // }
    clock_gettime(CLOCK_MONOTONIC, &begin);
    checkCudaErrors(secureCudaMemcpy(ecpreg, d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, encrypt , encrypt));
    checkCudaErrors(secureCudaMemcpy(ecpreg, d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, encrypt , encrypt));



    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // create and start timer
    // printf("Computing result using CUBLAS...");

    // execute the kernel
    // int nIter = 30;

    // CUBLAS version 2.0
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasHandle_t handle;
        cudaEvent_t compute_end, compute_start;

        checkCudaErrors(cublasCreate(&handle));

        // //Perform warmup operation with cublas
        // checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));

        // Allocate CUDA events that we'll use for timing
        checkCudaErrors(cudaEventCreate(&compute_start));
        checkCudaErrors(cudaEventCreate(&compute_end));

        cudaEventRecord(compute_start,NULL);
        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
        }
        cudaEventRecord(compute_end,NULL);
        checkCudaErrors(cudaEventSynchronize(compute_end));
        
        // if(encrypt)encrypt_gpu(d_C,mem_size_C/sizeof(int));
        // // decrypt_gpu(d_C,mem_size_C/sizeof(int));
	    // checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));
        // if(encrypt)decrypt_cpu(h_CUBLAS,mem_size_C/sizeof(int));

        checkCudaErrors(secureCudaMemcpy(ecpreg, h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost, encrypt, encrypt));

        printf("done.\n");

        // Record the stop event
        checkCudaErrors(cudaEventRecord(stop, NULL));

        // Wait for the stop event to complete
        checkCudaErrors(cudaEventSynchronize(stop));

        float msecTotal = 0.0f,msecCompute=0.0f;
        checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
        cudaEventElapsedTime(&msecCompute,compute_start,compute_end);
        // Compute and print the performance
        float msecPerMatrixMul = msecTotal/nIter;
        double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
        double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
        printf(
            "Performance= %.2f GFlop/s,Compute time= %.3f msec, Total time= %.3f msec, Size= %.0f Ops, nIter=%d\n",
            gigaFlops,
            msecCompute,
            msecTotal,
            flopsPerMatrixMul,
            nIter);

        // Destroy the handle
        checkCudaErrors(cublasDestroy(handle));
    }
    bool resCUBLAS;
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Total Time in CPU end: %lf s\n",((double)end.tv_sec - begin.tv_sec + 0.000000001 * (end.tv_nsec - begin.tv_nsec)));
    if(verify){
        // compute reference solution
        printf("Computing result using host CPU...");
        float *reference = (float *)malloc(mem_size_C);
        matrixMulCPU(reference, h_A2, h_B2, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
        printf("done.\n");

        // check result (CUBLAS)
        resCUBLAS = sdkCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f);

        if (resCUBLAS != true)
        {
            printDiff(reference, h_CUBLAS, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
        }

        free(reference);
        printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");
    }
    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

    if (resCUBLAS == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}

// ////////////////////////////////////////////////////////////////////////////////
// //! Secure CUDA memory copy
// //! @param ecpreg     Global Enchecap context
// //! @param dst        Destination memory address
// //! @param src        Source memory address
// //! @param count      Size in bytes to copy
// //! @param kind       Type of transfer
// //! @param encrypt_s  Whether to encrypt before copying data from the source or not (1 or 0)
// //! @param decrypt_d  Whether to decrypt after data's arrival at the destination or not (1 or 0)
// ////////////////////////////////////////////////////////////////////////////////
// cudaError_t secureCudaMemcpy(ECPreg ecpreg, void *dst, void *src, size_t count, enum cudaMemcpyKind kind, int encrypt_s, int decrypt_d)
// {
//     unsigned long eid = ecpreg.eid;
//     printf("\nsecureCudaMemcpying...\n");
//     cudaError_t ret;
//     if(kind == cudaMemcpyHostToDevice && encrypt_s)
//     {
//         printf("Before encryption in CPU: ** %u **\n", ((unsigned int*)src)[1]);
//         enclave_encrypt_cpu(eid, src, count/sizeof(int));
//         // encrypt_cpu(src, count, eid);
//         printf("After encryption in CPU: ** %u **\n", ((unsigned int*)src)[1]);
//     }
//     else if(kind == cudaMemcpyDeviceToHost && encrypt_s)
//         encrypt_gpu(src, count/sizeof(int));

//     ret = cudaMemcpy(dst, src, count, kind);
    
//     if(kind == cudaMemcpyHostToDevice && decrypt_d)
//         decrypt_gpu(dst, count/sizeof(int));
//     else if(kind == cudaMemcpyDeviceToHost && decrypt_d)
//     {
//         printf("Before decryption in CPU: ** %u **\n", ((unsigned int*)dst)[1]);
//         enclave_decrypt_cpu(eid, dst, count/sizeof(int));
//         // decrypt_cpu(dst, count, eid);
//         printf("After decryption in CPU: ** %u **\n", ((unsigned int*)dst)[1]);
//     }
//     printf("secureCudaMemcpying successfully!\n");
//     return ret;
// }

/* Entrance */
int beginMatrixMulCUBLAS(int argc, char **argv, ECPreg ecpreg)
{
    printf("[Matrix Multiply CUBLAS] - Starting...\n");

    int devID = 0, sizeMult = 5;
    sMatrixSize matrix_size;

    initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

    int matrix_result = matrixMultiply(argc, argv, devID, matrix_size, ecpreg);

    return matrix_result;
}
