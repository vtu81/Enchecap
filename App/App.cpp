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
#include "Enchecap.h"

/* Global EID shared by multiple threads */
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

    ECPreg ctx;
    if(initEnchecap(global_eid, ctx) == -1)
    {
        printf("Error while initiating Enchecap!\n");
        return -1;
    }
    printf("Successfully initialize Enchecap!\neid = %ld\n", global_eid);
 
    /* Utilize edger8r attributes */
    // edger8r_array_attributes();
    // edger8r_pointer_attributes(); //print: Checksum(0x0x7ffdbe8d31d0, 100) = 0xfffd4143
    // edger8r_type_attributes();
    // edger8r_function_attributes();
    
    /* Utilize trusted libraries */
    // ecall_libc_functions();
    // ecall_libcxx_functions();
    // ecall_thread_functions();

    /* Hello world! */
    ecall_print_helloworld(global_eid);

    beginMatrixMulCUBLAS(argc, argv, global_eid);

    /* Destroy the enclave */
    sgx_destroy_enclave(global_eid);
    
    // printf("Info: SampleEnclave successfully returned.\n");

    // printf("Enter a character before exit ...\n");
    // getchar();

    callGPU();

    return 0;
}

