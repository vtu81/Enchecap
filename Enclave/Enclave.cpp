#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
// #include <time.h>
// #include <omp.h>
#include "sgx_trts.h"
#include "sgx_tseal.h"
#include "string.h"

/**
 * printf: 
 *   Invokes OCALL to display the enclave buffer to the terminal.
 */
int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

void ecall_print_helloworld()
{
    printf("Hello world!\n");
}

/**
 * ecall_encrypt_cpu():
 *   encrypt `data` with length `len` in the enclave
 * 
 * Usage:
 *   should only be used for encrypting **user's keys** with **GPU-generated public key** from `ECPreg reg.userGpuPublicKey`.
 */
void ecall_encrypt_cpu(void* data, int len, void** en_keys)
{
	printf("@ENCLAVE: \"CPU encrypting in enclave...");
	printf("** %u ** -> ", ((unsigned int*)data)[1]);
	
    unsigned long int n, e;
	n = ((unsigned long int*)*en_keys)[0];
	e = ((unsigned long int*)*en_keys)[1];
	printf("n = %u, e = %u\n", n, e);

    // struct timespec __begin,__end;
    // clock_gettime(CLOCK_MONOTONIC, &__begin);
	unsigned int *mm = (unsigned int *)data,*en = mm;
	#ifdef _DEBUG
	printf("n%u\n\n\n\n",n);
	#endif
	// #pragma omp parallel for
	for(int i = 0; i < len; i++){
		unsigned long key = e, k = 1, exp = mm[i] % n;
		while(key){
			if(key % 2){
				k *= exp;
				k %= n;
			}
			key /= 2;
			exp *= exp;
			exp %= n;
		}
		en[i] = (unsigned int)k;		
	}
	// clock_gettime(CLOCK_MONOTONIC,&__end);
	// printf("CPU Encryption in enclave:%lf\n",((double)__end.tv_sec - __begin.tv_sec + 0.000000001 * (__end.tv_nsec - __begin.tv_nsec)));
	printf("** %u **", ((unsigned int*)data)[1]);
	printf(" successfully!\"\n");
}

/**
 * ecall_decrypt_cpu():
 *   decrypt `data` with length `len` in the enclave
 * 
 * Usage:
 *   Used for testing; not necessary in practical cases.
 */
void ecall_decrypt_cpu(void* data, int len, void** sgx_keys)
{
	printf("@ENCLAVE: \"CPU decrypting in enclave...");
	printf("** %u ** -> ", ((unsigned int*)data)[1]);

	unsigned long int ptrr[3];
	unseal((sgx_sealed_data_t*)*sgx_keys, sizeof(sgx_sealed_data_t) + sizeof(ptrr), (uint8_t*)&ptrr, 3*sizeof(unsigned long int));

    unsigned long int n, d;
	n = ptrr[1];
	d = ptrr[0];

    // struct timespec __begin,__end;
    // clock_gettime(CLOCK_MONOTONIC, &__begin);
	unsigned int *mm=(unsigned int *)data,*en=mm;
	// #pragma omp parallel for
	for(int i=0;i<len;i++) {
		unsigned long ct = en[i]%n,k = 1,key=d;
		while(key){
			if(key%2==1){
				k*=ct;
				k%=n;
			}
			key/=2;
			ct*=ct;
			ct%=n;
		}
		mm[i] = (unsigned int)k;
	}
	// clock_gettime(CLOCK_MONOTONIC,&__end);
	// printf("CPU Decryption in enclave:%lf\n",((double)__end.tv_sec - __begin.tv_sec + 0.000000001 * (__end.tv_nsec - __begin.tv_nsec)));
	printf("** %u **", ((unsigned int*)data)[1]);
	printf(" successfully!\"\n");
}

/**
 * ecall_generate_keys_straw_man():
 *   Generate and seal SGX's keys.
 *   Let `*sgx_keys_addr` = the sealed keys' address, and `sgx_pub_key_addr` = the unsealed pub key (n, e) address (in host memory)
 */
void ecall_generate_keys_straw_man(void** sgx_keys_addr, void** sgx_pub_key_addr)
{
	unsigned long int p, q, n, e, d;
	p = 74531;
    q = 37019;
    e = 0x10001;
	d = 985968293;
	n = p * q;

	unsigned long int ptr[3] = {d, n, e};
	printf("User keys (d, n, e): %u %u %u\n", ptr[0], ptr[1], ptr[2]);

	size_t sealed_size = sizeof(sgx_sealed_data_t) + sizeof(ptr);
	uint8_t* sealed_data = (uint8_t*)malloc(sealed_size);
	seal((uint8_t*)&ptr, 3*sizeof(unsigned long int), (sgx_sealed_data_t*)sealed_data, sealed_size);
	printf("sealed_data address: %p\n", sealed_data);
	*sgx_keys_addr = sealed_data;
	printf("sizeof(sealed_data) = %d\n",sizeof(sealed_data));

	unsigned long int* pub_key_addr = (unsigned long int*)malloc(sizeof(unsigned long int) * 2);
	pub_key_addr[0] = n;
	pub_key_addr[1] = e;
	*sgx_pub_key_addr = pub_key_addr;

	printf("@ENCLAVE: Get and seal user keys successfully!\n");
}

/**
 * test():
 *   Peek into the sealed structure
 * 
 * Usage:
 *   Only for debug!
 */
void test(void** user_keys)
{
	printf("sizeof(int) = %d\n", sizeof(int));
	unsigned long int ptrr[3];
	unseal((sgx_sealed_data_t*)*user_keys, sizeof(sgx_sealed_data_t) + sizeof(ptrr), (uint8_t*)&ptrr, 3*sizeof(unsigned long int));
	printf("(test)unsealed data content: %u %u %u\n", ptrr[0], ptrr[1], ptrr[2]);
}

/**
 * @brief      Seals the plaintext given into the sgx_sealed_data_t structure
 *             given.
 *
 * @details    The plaintext can be any data. uint8_t is used to represent a
 *             byte. The sealed size can be determined by computing
 *             sizeof(sgx_sealed_data_t) + plaintext_len, since it is using
 *             AES-GCM which preserves length of plaintext. The size needs to be
 *             specified, otherwise SGX will assume the size to be just
 *             sizeof(sgx_sealed_data_t), not taking into account the sealed
 *             payload.
 *
 * @param      plaintext      The data to be sealed
 * @param[in]  plaintext_len  The plaintext length
 * @param      sealed_data    The pointer to the sealed data structure
 * @param[in]  sealed_size    The size of the sealed data structure supplied
 *
 * @return     Truthy if seal successful, falsy otherwise.
 */
sgx_status_t seal(uint8_t* plaintext, size_t plaintext_len, sgx_sealed_data_t* sealed_data, size_t sealed_size) {
    sgx_status_t status = sgx_seal_data(0, NULL, plaintext_len, plaintext, sealed_size, sealed_data);
    return status;
}

/**
 * @brief      Unseal the sealed_data given into c-string
 *
 * @details    The resulting plaintext is of type uint8_t to represent a byte.
 *             The sizes/length of pointers need to be specified, otherwise SGX
 *             will assume a count of 1 for all pointers.
 *
 * @param      sealed_data        The sealed data
 * @param[in]  sealed_size        The size of the sealed data
 * @param      plaintext          A pointer to buffer to store the plaintext
 * @param[in]  plaintext_max_len  The size of buffer prepared to store the
 *                                plaintext
 *
 * @return     Truthy if unseal successful, falsy otherwise.
 */
sgx_status_t unseal(sgx_sealed_data_t* sealed_data, size_t sealed_size, uint8_t* plaintext, uint32_t plaintext_len) {
    sgx_status_t status = sgx_unseal_data(sealed_data, NULL, NULL, (uint8_t*)plaintext, &plaintext_len);
    return status;
}

/**
 * ecall_encrypt_with_untrusted_key_cpu():
 *   unseal `sgx_user_keys` to get the user's keys, and encrypt the keys
 *   in the enclave using `untrusted_keys` --> (n, e), writing the result's addres into `*ret`.
 * 
 * Usage:
 *   should only be used for encrypting **user's keys** with **GPU-generated public key** from `ECPreg reg.userGpuPublicKey`.
 */
void ecall_encrypt_user_keys_with_untrusted_key_cpu(void** sgx_user_keys, void* untrusted_keys, void** ret)
{
	//unseal user's keys to `ptrr`
	unsigned long int ptrr[3];
	unseal((sgx_sealed_data_t*)*sgx_user_keys, sizeof(sgx_sealed_data_t) + sizeof(ptrr), (uint8_t*)&ptrr, 3*sizeof(unsigned long int));
	
	//save the user's keys in `data`, waiting for encryption
	void* data = malloc(3 * sizeof(unsigned long int));
	int len = 3 * sizeof(unsigned long int) / sizeof(int);
	((unsigned long int*)data)[0] = ptrr[0];
	((unsigned long int*)data)[1] = ptrr[1];
	((unsigned long int*)data)[2] = ptrr[2];

	printf("@ENCLAVE: \"before encryption, user's keys look like this\tdata[0] = %u, data[1] = %u, data[2] = %u\"\n", ((unsigned long int*)data)[0], ((unsigned long int*)data)[1], ((unsigned long int*)data)[2]);

	unsigned long int n, e, d;
	n = ((unsigned long long*)untrusted_keys)[0];
	e = ((unsigned long long*)untrusted_keys)[1];

	unsigned int *mm = (unsigned int *)data,*en = mm;
	for(int i = 0; i < len; i++){
		unsigned long key = e, k = 1, exp = mm[i] % n;
		while(key){
			if(key % 2){
				k *= exp;
				k %= n;
			}
			key /= 2;
			exp *= exp;
			exp %= n;
		}
		en[i] = (unsigned int)k;		
	}

	printf("@ENCLAVE: \"after encryption, user's keys look like this\tdata[0] = %u, data[1] = %u, data[2] = %u\"\n", ((unsigned long int*)data)[0], ((unsigned long int*)data)[1], ((unsigned long int*)data)[2]);
	*ret = data;

	// Debug only
	// d = 107433065;
	// #pragma omp parallel for
	// for(int i=0;i<len;i++) {
	// 	unsigned long ct = en[i]%n,k = 1,key=d;
	// 	while(key){
	// 		if(key%2==1){
	// 			k*=ct;
	// 			k%=n;
	// 		}
	// 		key/=2;
	// 		ct*=ct;
	// 		ct%=n;
	// 	}
	// 	mm[i] = (unsigned int)k;
	// }
	// printf("@ENCLAVE: \"after decryption, user's keys look like this\tdata[0] = %u, data[1] = %u, data[2] = %u\"\n", ((unsigned long int*)data)[0], ((unsigned long int*)data)[1], ((unsigned long int*)data)[2]);
}
