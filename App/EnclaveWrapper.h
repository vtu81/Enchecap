#include "Enchecap_common.h"

void enclave_generate_keys(ECPreg *ecpreg);
void* enclave_encrypt_user_keys_with_gpu_keys(ECPreg *ecpreg);
void enclave_encrypt_cpu(unsigned long eid, void* data, int len, void* sgx_user_keys);
void enclave_decrypt_cpu(unsigned long eid, void* data, int len, void* sgx_user_keys);
