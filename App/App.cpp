#include <stdio.h>
#include <string.h>
#include <assert.h>

# include <unistd.h>
# include <pwd.h>
# define MAX_PATH FILENAME_MAX

#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"
#include "DeviceApp.h"
#include "matrixMulCUBLAS.h"
#include "ErrorSGX.h"
#include "Enchecap_host.h"

/* Global EID shared by multiple threads; SHOULD NOT BE REMOVED!!! */
sgx_enclave_id_t global_eid = 0;

/* OCall functions */
void ocall_print_string(const char *str)
{
    /* Proxy/Bridge will check the length and null-terminate 
     * the input string to prevent buffer overflow. 
     */
    printf("%s", str);
}


/* Application entry */
int SGX_CDECL main(int argc, char *argv[])
{
    (void)(argc);
    (void)(argv);

    /**
     * Use a global `ECPreg ecpreg` to pass context of Enchecap
     */
    ECPreg ecpreg;
    printf("user_prime_pointer: %p\n", ecpreg.user_prime_pointer);
    if(initEnchecap(global_eid, &ecpreg) == -1) /* initialize ecpreg */
    {
        printf("Error while initiating Enchecap!\n");
        return -1;
    }
    printf("Successfully initialize Enchecap!\necp.eid = %ld\n", ecpreg.eid);
    printf("ecp.prime_pointer: %p\n", ecpreg.prime_pointer);
    printf("ecp.user_prime_pointer: %p\n", ecpreg.user_prime_pointer);
    

    ////////////////////////////////////////////////////////////////////////////////
    // Major code begin
    ////////////////////////////////////////////////////////////////////////////////
    
    /* Hello world! */
    ecall_print_helloworld(global_eid);
    /* CUBLAS */
    beginMatrixMulCUBLAS(argc, argv, ecpreg);
    /* simple GPU-calling demo */
    callGPU();

    ////////////////////////////////////////////////////////////////////////////////
    // Major code end
    ////////////////////////////////////////////////////////////////////////////////

    /* Destroy the enclave */
    sgx_destroy_enclave(ecpreg.eid);
    printf("Info: Successfully returned.\n");

    return 0;
}

