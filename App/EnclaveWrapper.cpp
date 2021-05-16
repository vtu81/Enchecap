#include "EnclaveWrapper.h"
#include "Enclave_u.h"
#include <stdio.h>
#include "ErrorSGX.h"

void print_error_message_(sgx_status_t ret);

void enclave_get_user_keys(ECPreg *ecpreg)
{
	ecall_get_user_key_straw_man(ecpreg->eid, &(ecpreg->sgx_user_keys));
	printf("ecpreg->sgx_user_keys: %p\n", ecpreg->sgx_user_keys);
	test(ecpreg->eid, &(ecpreg->sgx_user_keys));
}

void* enclave_encrypt_user_keys_with_gpu_keys(ECPreg *ecpreg)
{
	void* ret;
	ecall_encrypt_user_keys_with_untrusted_key_cpu(ecpreg->eid, &(ecpreg->sgx_user_keys), ecpreg->cpu_gpu_keys, &ret);
	printf("In host memory, encrypted user's keys now look like this\tret[0] = %u, ret[1] = %u, ret[2] = %u\n", ((unsigned long int*)ret)[0], ((unsigned long int*)ret)[1], ((unsigned long int*)ret)[2]);
	return ret;
}

void enclave_encrypt_cpu(unsigned long eid, void* data, int len, void* sgx_user_keys,unsigned *auxBuf)
{
	sgx_status_t ret;

	printf("sgx_user_keys: %p\n", sgx_user_keys);
	ret = ecall_encrypt_cpu(eid, data, len, &sgx_user_keys,auxBuf);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}

void enclave_decrypt_cpu(unsigned long eid, void* data, int len, void* sgx_user_keys,unsigned *auxBuf)
{
	sgx_status_t ret;

	ret = ecall_decrypt_cpu(eid, data, len, &sgx_user_keys,auxBuf);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}