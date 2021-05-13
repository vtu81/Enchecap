#include "Enchecap_host.h"

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
    /* Fetch the user's key and store them in the enclave */
    //......

    /* Initialize GPU key */
    unsigned long long *pk;
    unsigned long long *prime_pointer;
    pk=(unsigned long long *)malloc(sizeof(unsigned long long)*3);
    // cudaGetPublicKey(pk);
    cudaGetPublicKeyStrawMan(pk, &prime_pointer); // FIXME: Just a straw man; please replace it with the real one without changing the interface.
    ecpreg->gpuPublicKey = (void*)pk;
    ecpreg->prime_pointer = (void*)prime_pointer;

    /* Key exchange */
    //FIXME: Not done yet. The GPU address `user_prime_pointer` should eventually point to the user's keys (d, n, e).
    ecpreg->user_prime_pointer = ecpreg->prime_pointer;
    
    /* Initialize the enclave */
    if(initialize_enclave(eid) < 0){
        printf("Enter a character before exit ...\n");
        getchar();
        return -1; 
    }
    ecpreg->eid = eid; // saved `eid` into ECPreg
    return 0;
}
