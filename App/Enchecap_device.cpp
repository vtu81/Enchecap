#include "Enchecap_device.h"

////////////////////////////////////////////////////////////////////////////////
//! Secure CUDA memory copy
//! @param ecpreg     Global Enchecap context
//! @param dst        Destination memory address
//! @param src        Source memory address
//! @param count      Size in bytes to copy
//! @param kind       Type of transfer
//! @param encrypt_s  Whether to encrypt before copying data from the source or not (1 or 0)
//! @param decrypt_d  Whether to decrypt after data's arrival at the destination or not (1 or 0)
////////////////////////////////////////////////////////////////////////////////
cudaError_t secureCudaMemcpy(ECPreg ecpreg, void *dst, void *src, size_t count, enum cudaMemcpyKind kind, int encrypt_s, int decrypt_d)
{
    unsigned long eid = ecpreg.eid;
    printf("\nsecureCudaMemcpying...\n");
    cudaError_t ret;
    if(kind == cudaMemcpyHostToDevice && encrypt_s)
    {
        printf("Before encryption in CPU: ** %u **\n", ((unsigned int*)src)[1]);
        enclave_encrypt_cpu(eid, src, count/sizeof(int), ecpreg.shared_gpu_pub_key_host);
        printf("After encryption in CPU: ** %u **\n", ((unsigned int*)src)[1]);
    }
    else if(kind == cudaMemcpyDeviceToHost && encrypt_s)
    {
        encrypt_gpu(src, count/sizeof(int), ecpreg.shared_sgx_pub_key_device);
    }        
    
    ret = cudaMemcpy(dst, src, count, kind);
    
    if(kind == cudaMemcpyHostToDevice && decrypt_d)
    {
        decrypt_gpu(dst, count/sizeof(int), ecpreg.gpu_keys);
    }
    else if(kind == cudaMemcpyDeviceToHost && decrypt_d)
    {
        printf("Before decryption in CPU: ** %u **\n", ((unsigned int*)dst)[1]);
        enclave_decrypt_cpu(eid, dst, count/sizeof(int), ecpreg.sgx_keys);
        printf("After decryption in CPU: ** %u **\n", ((unsigned int*)dst)[1]);
    }
    printf("secureCudaMemcpying successfully!\n");
    return ret;
}