#include "EnclaveWrapper.h"
#include "Enclave_u.h"
#include <stdio.h>
#include "ErrorSGX.h"

void print_error_message_(sgx_status_t ret);

void enclave_generate_keys(ECPreg *ecpreg)
{
	ecpreg->shared_sgx_pub_key_host = (unsigned long int*)malloc(sizeof(unsigned long int) * 2);
	ecall_generate_keys_straw_man(ecpreg->eid, &(ecpreg->sgx_keys), ecpreg->shared_sgx_pub_key_host);
	printf("ecpreg->sgx_keys: %p\n", ecpreg->sgx_keys);
	printf("ecpreg->shared_sgx_pub_key_host: %u %u\n", ((unsigned long int*)(ecpreg->shared_sgx_pub_key_host))[0], ((unsigned long int*)(ecpreg->shared_sgx_pub_key_host))[1]);
	test(ecpreg->eid, &(ecpreg->sgx_keys));
}

void* enclave_encrypt_user_keys_with_gpu_keys(ECPreg *ecpreg)
{
	void* ret;
	ecall_encrypt_user_keys_with_untrusted_key_cpu(ecpreg->eid, &(ecpreg->sgx_user_keys), ecpreg->cpu_gpu_keys, &ret);
	printf("In host memory, encrypted user's keys now look like this\tret[0] = %u, ret[1] = %u, ret[2] = %u\n", ((unsigned long int*)ret)[0], ((unsigned long int*)ret)[1], ((unsigned long int*)ret)[2]);
	return ret;
}

void enclave_encrypt_cpu(unsigned long eid, void* data, int len, void* en_keys)
{
	sgx_status_t ret;

	printf("Keys used to encrypt data in enclave: %p\n", en_keys);
	ret = ecall_encrypt_cpu(eid, data, len, &en_keys);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}

void enclave_decrypt_cpu(unsigned long eid, void* data, int len, void* sgx_keys)
{
	sgx_status_t ret;

	ret = ecall_decrypt_cpu(eid, data, len, &sgx_keys);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}