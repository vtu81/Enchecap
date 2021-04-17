#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
// #include <time.h>
// #include <omp.h>

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

/** FIXME:
 * should accept the addresses of keys (n, e) as parameters in a safe way
 */
/**
 * ecall_encrypt_cpu():
 *   encrypt `data` with length `len` in the enclave
 * 
 * Usage:
 *   should only be used for encrypting **user's keys** with **GPU-generated public key** from `ECPreg reg.userGpuPublicKey`.
 */
void ecall_encrypt_cpu(void* data, int len, unsigned long int n, unsigned long int e)
{
	printf("@ENCLAVE: \"CPU encrypting in enclave...");
	printf("** %u ** -> ", ((unsigned int*)data)[1]);
    
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

/** FIXME:
 * should accept the addresses of keys (n, d) as parameters in a safe way
 */
/**
 * ecall_decrypt_cpu():
 *   decrypt `data` with length `len` in the enclave
 * 
 * Usage:
 *   Used for testing; not necessary in practical cases.
 */
void ecall_decrypt_cpu(void* data, int len, unsigned long int n, unsigned long int d)
{
	printf("@ENCLAVE: \"CPU decrypting in enclave...");
	printf("** %u ** -> ", ((unsigned int*)data)[1]);
    // unsigned long int n, d;
	// n = 74531 * 37019;
	// d = 985968293;

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