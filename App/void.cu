/******************************************************************
* pure CPU version 
* intermediate by-product, for testing the unsigned type  
* preparation for parallel execution
* developed by Shay P.C. 2021/3/24 
* ***************************************************************/
/******************************************************************
    - temporarily using clock() for seed
    usuage: use cudaMemcpy(&prime_host,prime_pointer,sizeof(prime_host),cudaMemcpyDeviceToHost)
    to retrieve the generated prime
 * ***************************************************************/
 #include <stdlib.h>
 #include <stdio.h>
 #define POOL 10
 #define _DEBUG 1
 // FIXME: margin
 // #define _fuck_cuda
 #ifndef _fuck_cuda
 #include <cuda_runtime.h>
 #else
 #define __global__
 #define __device__ 
 #define __shared__
 #define __constant__
 #endif
 static unsigned int small_primes_host[]={3,5,7,9,11,13,17,19,23,29,31,37,41,43,47,53,59};
 __constant__ unsigned int small_primes[]={3,5,7,9,11,13,17,19,23,29,31,37,41,43,47,53,59};
 __device__ int tt=1023141123;
 __constant__ int * _ret;
 __constant__ unsigned long long * prime_pointer;
 #define nprimes (sizeof(small_primes_host)/sizeof(unsigned))
 __device__ void prime(unsigned long long x,int *ret){
     // dirty implementation
     for(int i=59;1;i+=2){
         if(x%i==0) {*ret=0;return;}
         if(i*i>x) {*ret=1;return;}
     }
 }
 __device__ void test(unsigned long long psu_prime,int *ret){
     // target: use sieve to leave out 99.9% (speedup 1000)
     // TODO: paralell
     for(int i=0;i< nprimes ;i++)
         if(psu_prime%small_primes[i]==0){*ret=0;return;}
     {*ret=1;return;}
 }
 __device__ void random_p(unsigned long long *ret){
     __shared__ int cnt,candidates[POOL+1];
     // __shared__ int* _ret;
     #ifdef _fuck_cuda
     _ret=(int*)malloc(sizeof(int));
     #else 
     #endif
     int t=tt*2+1;
     do{
         cnt=0;
         for(int i=t;cnt<POOL;i+=2){
             test(i,_ret);
             __syncthreads();
             if(*_ret)candidates[cnt++]=i;
         }
         for(int i=0;i<cnt;i++){
             prime(candidates[i],_ret);
             __syncthreads();
             if(*_ret){*ret=candidates[i];return;}
 
         }
         t=candidates[cnt-1]+2;
     }
     while(1);
     {*ret=-1;return;}
 }
 __global__ void RSA_key_generator(unsigned long long *pointer){
     // tt=1023141123;
     // random_p(pointer);
     // printf("from device :%lld\n",*pointer);
     // tt=30;
     // random_p(pointer);
     // printf("from device : prime 2=%lld\n",*pointer);
     for(int i=0;i<2;i++){
         tt=clock();
         random_p(pointer);
         printf("from device :%llu\n",*pointer);
     }
     // seed=time()
 }
 #ifdef _DEBUG
 int main(void){
     unsigned long long prime_host;
     // int64_t *pointer=NULL;
     #ifdef _fuck_cuda
     RSA_key_generator(prime_pointer);
     #else
     int *ret_device;
     cudaMalloc(&ret_device,sizeof(int));
     cudaMalloc(&prime_pointer,sizeof(unsigned long long));
     cudaMemcpyToSymbol(_ret,&ret_device,sizeof(ret_device));
     RSA_key_generator<<<1,1>>>(prime_pointer);
     cudaDeviceSynchronize();
     // cudaMemcpyFromSymbol(&prime_host,prime_pointer,sizeof(prime_host));
     cudaMemcpy(&prime_host,prime_pointer,sizeof(prime_host),cudaMemcpyDeviceToHost);
     #endif
     printf("from host :%llu",prime_host);
     return 0;
 }
 #endif