#ifndef ENCHECAP_H__
#define ENCHECAP_H__

/* Registers structure for necessary address:
 * 1. in the enclave
 * 2. in device memory
 **/
typedef struct ECPreg
{
    void* userPrivateKey;
    void* userPublicKey;
    void* userGpuPublicKey;
    
    void* gpuPrivateKey;
    void* gpuPublicKey;
    void* gpuUserPublicKey;
    void* gpuUserPrivateKey;
    
} ECPreg;

int initEnchecap(unsigned long &eid, ECPreg ecpreg);

#endif