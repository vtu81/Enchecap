/******************************************************************
 * pure CPU version 
 * intermediate by-product, for testing the unsigned type  
 * preparation for parallel execution
 * developed by Shay P.C. 2021/3/24
 * ***************************************************************/
 #include <stdlib.h>
 #include <stdio.h>
 #define POOL 10
 // #define _DEBUG 1
 // FIXME: margin
 #include <cuda_runtime.h>
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
 __device__ void swap(long long *a,long long *b){
     long long t;
     t=*a;
     *a=*b;
     *b=t;
 }
 __device__ void gcd(unsigned long long a,unsigned long long b,unsigned long long *ret){
     long long k[2][2];
     unsigned long long arc[2];
     // assert(a>b);
     if(b>a)swap((long long *)&a,(long long*)&b);
     printf("%llu %llu\n",a,b);
     arc[0]=a;arc[1]=b;
     k[0][0]=k[1][1]=1;
     k[1][0]=k[0][1]=0;
     while(b!=0){
         for(int i=0;i<2;i++){
             k[0][i]-=a/b*k[1][i];
         }
         // while(k[0][0]<0)        {k[0][0]+=arc[1];k[0][1]-=arc[0];}
         // while(k[0][0]>=arc[1])  {k[0][0]-=arc[1];k[0][1]+=arc[0];}
 
         a=a%b;
         swap((long long *)&a,(long long *)&b);
         for(int i=0;i<2;i++)swap(&k[0][i],&k[1][i]);
     }
     printf("%d\n",a);
     printf("coeffs: %lld %lld \n",k[0][0],k[0][1]);
     printf("%lld\n",k[0][0]*arc[0]+k[0][1]*arc[1]);
     *ret=(unsigned long long)k[0][0];
 }
 __global__ void RSA_key_generator(unsigned long long *pointer){
     // // tt=1023141123;
     // for(int i=0;i<2;i++){
     //     tt=clock();
     //     // if(i==0)tt=30;
     //     // else tt=31;
     //     random_p(pointer+i);
     //     printf("from device :%llu\n",pointer[i]);
     // }
     // // __syncthreads();
     // gcd(pointer[0],pointer[1],pointer+2);
     // printf("%llu\n",pointer[2]);
     // // seed=time()
 
     unsigned long long p[2];    
     for(int i=0;i<2;i++){
         tt=clock();
         random_p(pointer+i);
         p[i]=pointer[i];
         printf("from device :%llu\n",pointer[i]);
     }
     if(p[0]>p[1])
         tt=p[0]/2+1;
     else tt=p[1]/2+1;
     pointer[1]=(p[0]-1)*(p[1]-1);
     random_p(pointer);
     // pointer[0] is now relatively prime to (p-1)(q-1)
     gcd(pointer[0],pointer[1],pointer+2);
 }
 #ifdef _DEBUG
 int main(void){
     unsigned long long prime_host;
     // int64_t *pointer=NULL;
     int *ret_device;
     cudaMalloc(&ret_device,sizeof(int));
     cudaMalloc(&prime_pointer,3*sizeof(unsigned long long));
     cudaMemcpyToSymbol(_ret,&ret_device,sizeof(ret_device));
     RSA_key_generator<<<1,1>>>(prime_pointer);
     cudaDeviceSynchronize();
     // cudaMemcpyFromSymbol(&prime_host,prime_pointer,sizeof(prime_host));
     cudaMemcpy(&prime_host,prime_pointer,sizeof(prime_host),cudaMemcpyDeviceToHost);
     printf("from host :%llu",prime_host);
     return 0;
 }
 #else 
 #include "DeviceApp.h"
 ////////////////////////////////////////////////////////////////////////////////
 //! init a transfer of GPU RSA public key
 //! pk is an array containing 3 64-bit unsigned int d, n and e accordingly
 //! whereas encrypt function being f(m)= m^d mod n, encrypt function g(m)=m^e mod n
 //! user should apply for mem space beforehand
 //! @param pk         des
 ////////////////////////////////////////////////////////////////////////////////
 void cudaGetPublicKey(unsigned long long *pk){
     unsigned long long prime_host[3];
     // int64_t *pointer=NULL;
     int *ret_device;
     cudaMalloc(&ret_device,sizeof(int));
     cudaMalloc(&prime_pointer,3*sizeof(unsigned long long));
     cudaMemcpyToSymbol(_ret,&ret_device,sizeof(ret_device));
     RSA_key_generator<<<1,1>>>(prime_pointer);
     cudaDeviceSynchronize();
     // cudaMemcpyFromSymbol(&prime_host,prime_pointer,sizeof(prime_host));
     cudaMemcpy(&prime_host,prime_pointer,sizeof(prime_host),cudaMemcpyDeviceToHost);
     printf("from host :%llu %llu %lld",prime_host[0],prime_host[1],prime_host[2]);
     for(int i=0;i<3;i++)pk[i]=prime_host[i];
 }

// FIXME
void cudaGetPublicKeyStrawMan(unsigned long long *cpu_gpu_keys, unsigned long long **gpu_gpu_keys_addr){
    unsigned long int p, q, n, e, d;
    p = 74531;
    q = 37019;
    e = 0x10001;
    d = 985968293;
    n = p * q;
    unsigned long long prime_host[3] = {d, n, e};

    cudaMalloc(&prime_pointer,3*sizeof(unsigned long long));

    cudaMemcpy(prime_pointer,&prime_host,sizeof(prime_host),cudaMemcpyHostToDevice);
    
    // unsigned long long test[3];
    // cudaMemcpy(&test,prime_pointer,sizeof(test),cudaMemcpyDeviceToHost);
    // printf("prime_host(cpu) :%llu %llu %lld\n",prime_host[0],prime_host[1],prime_host[2]);
    // printf("prime_pointer(gpu) :%llu %llu %lld\n",test[0],test[1],test[2]);

    for(int i=0;i<3;i++)cpu_gpu_keys[i] = prime_host[i];
    *gpu_gpu_keys_addr = prime_pointer;
    // printf("prime_pointer: %p\t*gpu_gpu_keys_addr: %p\n", prime_pointer, *gpu_gpu_keys_addr);
}
 #endif