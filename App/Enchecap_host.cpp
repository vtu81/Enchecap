#include "Enchecap_host.h"
#include "EnclaveWrapper.h"

/* Initialize the enclave:
 *   Call sgx_create_enclave to initialize an enclave instance
 */
int initialize_enclave(unsigned long &eid)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;
    
    /* Call sgx_create_enclave to initialize an enclave instance */
    /* Debug Support: set 2nd parameter to 1 */
    ret = sgx_create_enclave(ENCLAVE_FILENAME, SGX_DEBUG_FLAG, NULL, NULL, &eid, NULL);
    if (ret != SGX_SUCCESS) {
        print_error_message(ret);
        return -1;
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize Enchecap
//! 1. Initialize an enclave and write its id to `eid`
//! 2. User send the RSA key pair directly into enclave (not done yet; save at address XXX)
//! 3. Randomly generate a RSA key pair on the GPU (save at address XXX) and send the public key to main memory
//! 4. Encrypt user's key pair using the GPU's public key and send the encrypted keys back to device memory,
//!    then decrypt them on GPU (save at address XXX)
//! @param eid        Must be `global_eid` defined in App.cpp
//! @param ecpreg     Global Enchecap context
////////////////////////////////////////////////////////////////////////////////
int initEnchecap(unsigned long &eid, ECPreg *ecpreg)
{
    /* Initialize the enclave */
    if(initialize_enclave(eid) < 0){
        printf("Enter a character before exit ...\n");
        getchar();
        return -1; 
    }
    ecpreg->eid = eid; // saved `eid` into ECPreg
    printf("eid = %lu\n", eid);
    printf("---------Create enclave successfully!---------\n");

    /* Fetch the user's key and store them in the enclave */
    enclave_get_user_keys(ecpreg);
    printf("----Get and seal user's keys successfully!----\n");

    /* Initialize GPU key */
    unsigned long long *cpu_gpu_keys;
    unsigned long long *gpu_gpu_keys;
    cpu_gpu_keys=(unsigned long long *)malloc(sizeof(unsigned long long)*3);
    cudaGetPublicKey(cpu_gpu_keys,&gpu_gpu_keys);
    // cudaGetPublicKeyStrawMan(cpu_gpu_keys, &gpu_gpu_keys); // FIXME: Just a straw man; please replace it with the real one without changing the interface.
    ecpreg->cpu_gpu_keys = (void*)cpu_gpu_keys;
    ecpreg->gpu_gpu_keys = (void*)gpu_gpu_keys;
    printf("-----GPU's keys initialized successfully!-----\n");

    /* Key exchange */
    void* encrypted_user_keys = enclave_encrypt_user_keys_with_gpu_keys(ecpreg); // encrypt the user's keys with 
                                                                                 // GPU's public keys (n, e) at 
                                                                                 // `ecpreg->cpu_gpu_keys`
    cudaDecryptUserKeys(encrypted_user_keys, ecpreg->gpu_gpu_keys, &ecpreg->gpu_user_keys); // GPU decrypts the `encrypted_user_keys` with its (d, n)
                                                                                            // and hold the decrypted keys in device memory (as a 
                                                                                            // global pointer), whose address is recorded at `ecpreg->gpu_user_keys`
    printf("----------Key exchange successfully!----------\n");

    return 0;
}
