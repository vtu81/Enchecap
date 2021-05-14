#include "EnclaveWrapper.h"
#include "Enclave_u.h"
#include <stdio.h>
#include "ErrorSGX.h"

void print_error_message_(sgx_status_t ret);

void enclave_getUserKey(ECPreg *ecpreg)
{
	ecall_get_user_key_straw_man(ecpreg->eid, &(ecpreg->sgx_user_keys));
	printf("ecpreg->sgx_user_keys: %p\n", ecpreg->sgx_user_keys);
	test(ecpreg->eid, &(ecpreg->sgx_user_keys));
}

void enclave_encrypt_cpu(unsigned long eid, void* data, int len, void* sgx_user_keys)
{
	sgx_status_t ret;

	printf("sgx_user_keys: %p\n", sgx_user_keys);
	ret = ecall_encrypt_cpu(eid, data, len, &sgx_user_keys);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}

void enclave_decrypt_cpu(unsigned long eid, void* data, int len, void* sgx_user_keys)
{
	sgx_status_t ret;

	ret = ecall_decrypt_cpu(eid, data, len, &sgx_user_keys);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}