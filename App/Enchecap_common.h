#ifndef ENCHECAP_COMMON_H__
#define ENCHECAP_COMMON_H__

/* Registers structure for necessary address:
 * 1. in the enclave
 * 2. in device memory
 **/
typedef struct ECPreg
{
    unsigned long eid;

    void* userPrivateKey;
    void* userPublicKey;
    void* userGpuPublicKey;
    
    void* gpuPrivateKey;
    void* gpuPublicKey;
    void* gpuUserPublicKey;
    void* gpuUserPrivateKey;

    void* prime_pointer; // __constant__ variable on GPU
    void* user_prime_pointer; // Another __constant__ variable on GPU; but temporarily use the same address as `prime_pointer`
    
} ECPreg;

#endif