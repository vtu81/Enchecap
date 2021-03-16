#include "EnclaveWrapper.h"
#include "Enclave_u.h"
#include <stdio.h>
#include "ErrorSGX.h"

void print_error_message_(sgx_status_t ret);

void enclave_encrypt_cpu(void* data, int len, unsigned long eid)
{
	sgx_status_t ret;
	ret = ecall_encrypt_cpu(eid, data, len);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}

void enclave_decrypt_cpu(void* data, int len, unsigned long eid)
{
	sgx_status_t ret;
	ret = ecall_decrypt_cpu(eid, data, len);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}