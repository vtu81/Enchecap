#ifndef ENCHECAP_COMMON_H__
#define ENCHECAP_COMMON_H__

/* Registers structure for necessary address:
 * 1. in the enclave
 * 2. in device memory
 **/
typedef struct ECPreg
{
    unsigned long eid;
    void* cpu_gpu_keys; // GPU's public key (n, e)
    void* gpu_gpu_keys; // __constant__ pointer to GPU's own keys (d, n, e) on GPU
    void* gpu_user_keys; // Another __constant__ pointer to user's keys on GPU; but temporarily use the same address as `gpu_gpu_keys`
    void* sgx_user_keys; // **sealed** user keys address
} ECPreg;

#endif