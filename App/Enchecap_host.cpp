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

    /* Initialize SGX key */
    enclave_generate_keys(ecpreg);
    printf("----Get and seal user's keys successfully!----\n");

    /* Initialize GPU key */
    unsigned long long *shared_gpu_pub_key_host;
    unsigned long long *gpu_keys;
    shared_gpu_pub_key_host=(unsigned long long *)malloc(sizeof(unsigned long long)*2);
    // cudaGetPublicKey(shared_gpu_pub_key_host);
    cudaGetPublicKeyStrawMan(shared_gpu_pub_key_host, &gpu_keys); // FIXME: Just a straw man; please replace it with the real one without changing the interface.
    ecpreg->shared_gpu_pub_key_host = (void*)shared_gpu_pub_key_host;
    ecpreg->gpu_keys = (void*)gpu_keys;
    printf("ecpreg->gpu_keys: %p\n", ecpreg->gpu_keys);
	printf("ecpreg->shared_gpu_pub_key_host: %u %u\n", ((unsigned long int*)(ecpreg->shared_gpu_pub_key_host))[0], ((unsigned long int*)(ecpreg->shared_gpu_pub_key_host))[1]);
    printf("-----GPU's keys initialized successfully!-----\n");

    /* Key exchange */
    cudaGetSGXKey(ecpreg->shared_sgx_pub_key_host, &ecpreg->shared_sgx_pub_key_device);
    // void* encrypted_user_keys = enclave_encrypt_user_keys_with_gpu_keys(ecpreg); // encrypt the user's keys with 
                                                                                 // GPU's public keys (n, e) at 
                                                                                 // `ecpreg->cpu_gpu_keys`
    // cudaDecryptUserKeys(encrypted_user_keys, ecpreg->gpu_gpu_keys, &ecpreg->gpu_user_keys); // GPU decrypts the `encrypted_user_keys` with its (d, n)
                                                                                            // and hold the decrypted keys in device memory (as a 
                                                                                            // global pointer), whose address is recorded at `ecpreg->gpu_user_keys`
    printf("----------Key exchange successfully!----------\n");

    return 0;
}
