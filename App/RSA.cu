#ifndef RSA_CU__
#define RSA_CU__

#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "RSA_kernel.cu"
#define BUZZ_SIZE 10002

unsigned long int p, q, n, e, d;
void encrypt_cpu(void *ptr,int size);
void decrypt_cpu(void *ptr,int size);
void encrypt_gpu(void *ptr,int size);
void decrypt_gpu(void *ptr,int size);
int threadsPerBlock = 1024;
int blocksPerGrid;
float time_encrypt_cpu, time_decrypt_cpu,time_encrypt_gpu,time_decrypt_gpu;
void encrypt_cpu(void *h_data,int len) {
	struct timespec __begin,__end;
    clock_gettime(CLOCK_MONOTONIC, &__begin);
	unsigned int *mm=(unsigned int *)h_data,*en=mm;
	#ifdef _DEBUG
	printf("n%u\n\n\n\n",n);
	#endif
	#pragma omp parallel for
	for(int i=0;i<len;i++){
		unsigned long key=e,k=1,exp=mm[i]%n;
		while(key){
			if(key%2){
				k*=exp;
				k%=n;
			}
			key/=2;
			exp*=exp;
			exp%=n;
		}
		en[i] = (unsigned int)k;		
	}
	clock_gettime(CLOCK_MONOTONIC,&__end);
	printf("CPU Encryption:%lf\n",((double)__end.tv_sec - __begin.tv_sec + 0.000000001 * (__end.tv_nsec - __begin.tv_nsec)));
}

void encrypt_gpu(void *d_data,int len) {
	cudaEvent_t start_encrypt, stop_encrypt;
	unsigned long int key = e;
	// cudaSetDevice(1);
	unsigned int *dev_num=(unsigned int *)d_data;
	unsigned long *dev_key, *dev_den;
	cudaMalloc((void **) &dev_key, sizeof(long int));
	cudaMalloc((void **) &dev_den, sizeof(long int));
	cudaMemcpy(dev_key, &key, sizeof(long int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_den, &n, sizeof(long int), cudaMemcpyHostToDevice);
	cudaEventCreate(&start_encrypt);
	cudaEventCreate(&stop_encrypt);
	cudaEventRecord(start_encrypt);
	blocksPerGrid=(len+threadsPerBlock-1)/threadsPerBlock;
	rsa<<<blocksPerGrid, threadsPerBlock>>>(dev_num,dev_key,dev_den,len);
	cudaEventRecord(stop_encrypt);
	cudaEventSynchronize(stop_encrypt);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&time_encrypt_gpu, start_encrypt, stop_encrypt);
	cudaFree(dev_key);
	cudaFree(dev_den);
	time_encrypt_gpu /= 1000;
	printf("GPU Encryption:%f\n", time_encrypt_gpu);

}

void decrypt_gpu(void *d_data,int len) {
	cudaEvent_t start_decrypt, stop_decrypt;
	unsigned long int key = d;
	// cudaSetDevice(1);
	unsigned int *dev_num=(unsigned int*)d_data;
	unsigned long *dev_key, *dev_den;
	cudaMalloc((void **) &dev_key, sizeof(long int));
	cudaMalloc((void **) &dev_den, sizeof(long int));
	cudaMemcpy(dev_key, &key, sizeof(long int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_den, &n, sizeof(long int), cudaMemcpyHostToDevice);

	cudaEventCreate(&start_decrypt);
	cudaEventCreate(&stop_decrypt);
	cudaEventRecord(start_decrypt);
	blocksPerGrid=(len+threadsPerBlock-1)/threadsPerBlock;
	rsa<<<blocksPerGrid, threadsPerBlock>>>(dev_num,dev_key,dev_den,len);
	cudaEventRecord(stop_decrypt);
	cudaEventSynchronize(stop_decrypt);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&time_decrypt_gpu, start_decrypt, stop_decrypt);

	cudaFree(dev_key);
	cudaFree(dev_den);
	
	time_decrypt_gpu /= 1000;
	printf("GPU Decryption:%f\n", time_decrypt_gpu);
}

void decrypt_cpu(void *h_data,int len) {
	struct timespec __begin,__end;
    clock_gettime(CLOCK_MONOTONIC, &__begin);
	unsigned int *mm=(unsigned int *)h_data,*en=mm;
	#pragma omp parallel for
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
	clock_gettime(CLOCK_MONOTONIC,&__end);
	printf("CPU Decryption:%lf\n",((double)__end.tv_sec - __begin.tv_sec + 0.000000001 * (__end.tv_nsec - __begin.tv_nsec)));
}

#endif