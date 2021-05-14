#include "Enchecap_common.h"

void enclave_getUserKey(ECPreg *ecpreg);
void enclave_encrypt_cpu(unsigned long eid, void* data, int len, void* sgx_user_keys);
void enclave_decrypt_cpu(unsigned long eid, void* data, int len, void* sgx_user_keys);
