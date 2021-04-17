#include "matrixMulCUBLAS.h"

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
__global__ void rsa(unsigned int * num,unsigned long int * key,unsigned long int * den,const int len) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i>=len)return;
	unsigned int temp;
	temp = mod_optimized(num[i], *key, *den);
	//temp = mod_optimized(num[i], *key, *den);
	atomicExch(&num[i], temp);
}
