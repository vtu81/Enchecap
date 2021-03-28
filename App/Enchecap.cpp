#include "Enchecap.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pwd.h>
#include "sgx_urts.h"
#include "Enclave_u.h"
#include "ErrorSGX.h"

# define MAX_PATH FILENAME_MAX
# define TOKEN_FILENAME   "enclave.token"
# define ENCLAVE_FILENAME "enclave.signed.so"

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

/* Initialize Enchecap
 * 1. Initialize an enclave and write its id to `eid`
 * 2. User send the RSA key pair directly into enclave (not done yet; save at address XXX)
 * 3. Randomly generate a RSA key pair on the GPU (save at address XXX) and send the public key to main memory
 * 4. Encrypt user's key pair using the GPU's public key and send the encrypted keys back to device memory,
 *    then decrypt them on GPU (save at address XXX)
**/
int initEnchecap(unsigned long &eid, ECPreg ecpreg)
{
    /* Initialize the enclave */
    if(initialize_enclave(eid) < 0){
        printf("Enter a character before exit ...\n");
        getchar();
        return -1; 
    }
    return 0;
}