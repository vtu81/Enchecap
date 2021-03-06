#include "DeviceRSA.h"

int threadsPerBlock = 1024;
int blocksPerGrid;
float time_encrypt_cpu, time_decrypt_cpu, time_encrypt_gpu, time_decrypt_gpu;
extern unsigned long int p, q, n, e, d; /* FIXME: Shared variables are bad; 
										 * Should be done in a invisible way wrapped inside Enchecap! */

/********************* CUDA Kernel Functions Begin *********************/
__device__ long long int mod(int base, int exponent, int den) {

	long long int ret;
	ret = 1;
	for (int i = 0; i < exponent; i++) {
		ret *= base;
		ret = ret % den;
	}
	return ret;

}

__device__ unsigned int mod_optimized(unsigned int base,unsigned long int exp,unsigned long int modulus){
	unsigned long p=1;
	unsigned long tmp=base;
	while(exp){
		tmp%=modulus;
		if(exp%2){
			p*=tmp;
			p%=modulus;
		}
		tmp=tmp*tmp;
		exp/=2;
	}
	return p;
}

__global__ void rsa_old(unsigned int * num,unsigned long int * key,unsigned long int * den,const int len) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=len)return;
	unsigned int temp;
	temp = mod_optimized(num[i], *key, *den);
	//temp = mod_optimized(num[i], *key, *den);
	atomicExch(&num[i], temp);
}

__global__ void rsa(unsigned int * num, unsigned long long *keys, const int eORd, const int len) {
	unsigned long int *key;
	unsigned long int *den;
	if(eORd == 1) //encrypt using (n, e)
	{
		key = (unsigned long int *)&keys[1];
		den = (unsigned long int *)&keys[0];
	}
	else //decrypt using (d, n, e)
	{
		key = (unsigned long int *)&keys[0];
		den = (unsigned long int *)&keys[1];
	}
	

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=len)return;
	unsigned int temp;
	temp = mod_optimized(num[i], *key, *den);
	//temp = mod_optimized(num[i], *key, *den);
	atomicExch(&num[i], temp);
}

/********************* CUDA Kernel Functions End *********************/
/* An encrypting function on GPU */
void encrypt_gpu(void *d_data, int len, void *en_keys) {
	cudaEvent_t start_encrypt, stop_encrypt;
	unsigned int *dev_num=(unsigned int*)d_data;
	cudaEventCreate(&start_encrypt);
	cudaEventCreate(&stop_encrypt);
	cudaEventRecord(start_encrypt);
	blocksPerGrid=(len+threadsPerBlock-1)/threadsPerBlock;
	rsa<<<blocksPerGrid, threadsPerBlock>>>(dev_num,(unsigned long long *)en_keys,1,len);
	cudaEventRecord(stop_encrypt);
	cudaEventSynchronize(stop_encrypt);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&time_encrypt_gpu, start_encrypt, stop_encrypt);

	time_encrypt_gpu /= 1000;
	printf("GPU Encryption:%f\n", time_encrypt_gpu);

}
/* An decrypting function on GPU */
void decrypt_gpu(void *d_data, int len, void *gpu_keys) {
	cudaEvent_t start_decrypt, stop_decrypt;
	unsigned int *dev_num=(unsigned int*)d_data;
	cudaEventCreate(&start_decrypt);
	cudaEventCreate(&stop_decrypt);
	cudaEventRecord(start_decrypt);
	blocksPerGrid=(len+threadsPerBlock-1)/threadsPerBlock;
	rsa<<<blocksPerGrid, threadsPerBlock>>>(dev_num,(unsigned long long *)gpu_keys,0,len);
	cudaEventRecord(stop_decrypt);
	cudaEventSynchronize(stop_decrypt);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&time_decrypt_gpu, start_decrypt, stop_decrypt);

	time_decrypt_gpu /= 1000;
	printf("GPU Decryption:%f\n", time_decrypt_gpu);
}




/** FIXME:
 * should accept another parameter `ECPreg ecpreg` and extract the address of keys (n, e) from it
 */
/* An encrypting function on GPU */
void encrypt_gpu_old(void *d_data, int len) {
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
	rsa_old<<<blocksPerGrid, threadsPerBlock>>>(dev_num,dev_key,dev_den,len);
	cudaEventRecord(stop_encrypt);
	cudaEventSynchronize(stop_encrypt);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&time_encrypt_gpu, start_encrypt, stop_encrypt);
	cudaFree(dev_key);
	cudaFree(dev_den);
	time_encrypt_gpu /= 1000;
	printf("GPU Encryption(deprecated):%f\n", time_encrypt_gpu);

}

/** FIXME:
 * should accept another parameter `ECPreg ecpreg` and extract the address of keys (n, d) from it
 */
/* An decrypting function on GPU */
void decrypt_gpu_old(void *d_data, int len) {
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
	rsa_old<<<blocksPerGrid, threadsPerBlock>>>(dev_num,dev_key,dev_den,len);
	cudaEventRecord(stop_decrypt);
	cudaEventSynchronize(stop_decrypt);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&time_decrypt_gpu, start_decrypt, stop_decrypt);

	cudaFree(dev_key);
	cudaFree(dev_den);
	
	time_decrypt_gpu /= 1000;
	printf("GPU Decryption(deprecated):%f\n", time_decrypt_gpu);
}

/* (Deprected) An unsafe encrypting function on CPU */
void encrypt_cpu_old(void *h_data, int len) {
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
	printf("CPU Encryption(deprecated):%lf\n",((double)__end.tv_sec - __begin.tv_sec + 0.000000001 * (__end.tv_nsec - __begin.tv_nsec)));
}

/* (Deprected) An unsafe decrypting function on CPU */
void decrypt_cpu_old(void *h_data, int len) {
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
	printf("CPU Decryption(deprecated):%lf\n",((double)__end.tv_sec - __begin.tv_sec + 0.000000001 * (__end.tv_nsec - __begin.tv_nsec)));
}
