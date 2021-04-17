#include "EnclaveWrapper.h"
#include "Enclave_u.h"
#include <stdio.h>
#include "ErrorSGX.h"

void print_error_message_(sgx_status_t ret);

/** FIXME:
 * should accept another parameter `ECPreg ecpreg` and extract the address of keys (n, e) from it
 */
void enclave_encrypt_cpu(unsigned long eid, void* data, int len)
{
	sgx_status_t ret;

	unsigned long int p, q, n, e, d;
	p = 74531;
    q = 37019;
    e = 0x10001;
	d = 985968293;
	n = p * q;

	ret = ecall_encrypt_cpu(eid, data, len, n, e);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}

/** FIXME:
 * should accept another parameter `ECPreg ecpreg` and extract the address of keys (n, d) from it
 */
void enclave_decrypt_cpu(unsigned long eid, void* data, int len)
{
	sgx_status_t ret;

	unsigned long int p, q, n, e, d;
	p = 74531;
    q = 37019;
    e = 0x10001;
	d = 985968293;
	n = p * q;
	
	ret = ecall_decrypt_cpu(eid, data, len, n, d);
	if (ret != SGX_SUCCESS) {
        print_error_message(ret);
	}
}