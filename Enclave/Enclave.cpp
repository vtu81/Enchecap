/*
 * Copyright (C) 2011-2020 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "Enclave.h"
#include "Enclave_t.h" /* print_string */
#include <stdarg.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
// #include <time.h>
// #include <omp.h>

/* 
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

void ecall_encrypt_cpu(void* data, int len)
{
	printf("@ENCLAVE: \"CPU encrypting in enclave...");
	printf("** %u ** -> ", ((unsigned int*)data)[1]);

    unsigned long int p, q, n, e, d;
    p = 74531;
    q = 37019;
    e = 0x10001;
	d = 985968293;
	n = p * q;
    
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

void ecall_decrypt_cpu(void* data, int len)
{
	printf("@ENCLAVE: \"CPU decrypting in enclave...");
	printf("** %u ** -> ", ((unsigned int*)data)[1]);
    unsigned long int p, q, n, e, d;
    p = 74531;
    q = 37019;
    e = 0x10001;
	d = 985968293;
	n = p * q;

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