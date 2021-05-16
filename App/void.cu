/******************************************************************
 * pure CPU version 
 * intermediate by-product, for testing the unsigned type  
 * preparation for parallel execution
 * developed by Shay P.C. 2021/3/24
 * ***************************************************************/
 #include <stdlib.h>
 #include <stdio.h>
 #define POOL 10
//  #define _DEBUG 1
 #define _DEBUG2 1
 #include <cuda_runtime.h>
 __constant__ unsigned int small_primes[]={3,5,7,9,11,13,17,19,23,29,31,37,41,43,47,53,59};
 __device__ unsigned int tt=1023141123;
 __constant__ int * _ret;
 __constant__ unsigned long long * prime_pointer;
 __constant__ unsigned long long * gpu_user_keys;
 #define nprimes (sizeof(small_primes)/sizeof(unsigned))
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
     {*ret=0;return;}
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
     printf("%llu\n",a);
     printf("coeffs: %lld %lld \n",k[0][0],k[0][1]);
     while(k[0][1]<0){
        k[0][1]+=arc[0];
        k[0][0]-=arc[1];
     }
     printf("%lld\n",k[0][0]*arc[0]+k[0][1]*arc[1]);
     *ret=(unsigned long long)k[0][1];
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
    //  p = 74831;
    //  q = 37619;
    //  e = 0x10001;
    //  d = 780225773;
    //  n = p * q;
      unsigned long long lcm;    
      static const unsigned long long upperBoundMask=(1<<24)-1,lowerBoundMask=1<<16;
     for(int i=0;i<2;i++){
         #ifdef _DEBUG2
         tt=i?37415:18809;
         #else 
         tt=(clock()&upperBoundMask)|lowerBoundMask;
         #endif
         random_p(pointer+i);
         printf("from device :%llu\n",pointer[i]);
     }
     if(pointer[0]>pointer[1])
         tt=pointer[0]/2+1;
     else tt=pointer[1]/2+1;
     lcm=(pointer[0]-1)*(pointer[1]-1);
     pointer[1]=pointer[0]*pointer[1];          // pointer[1]=N
     random_p(pointer+2);
     printf("from device :e=%llu\n",pointer[2]);// pointer[2] is now relatively prime to (p-1)(q-1)
     gcd(lcm,pointer[2],pointer);               // pointer[0]=d
    // e,N is the public key 
    // d,N is the private key 
 }
 #ifndef _DEBUG
 #include "DeviceRSA.h"
 #include "DeviceApp.h"
 #endif
////////////////////////////////////////////////////////////////////////////////
//! init a transfer of GPU RSA public key
//! pk is an array containing 3 64-bit unsigned int d, n and e accordingly
//! whereas encrypt function being f(m)= m^d mod n, encrypt function g(m)=m^e mod n
//! user should apply for mem space beforehand
//! @param pk         des
////////////////////////////////////////////////////////////////////////////////
void cudaGetPublicKey(unsigned long long *pk,unsigned long long **gpu_gpu_keys_addr){
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
    printf("from host :%llu %llu %llu\n",prime_host[0],prime_host[1],prime_host[2]);
    
    //  fit the  interface
    *gpu_gpu_keys_addr=prime_pointer;
    pk[0]=prime_host[1];
    pk[1]=prime_host[2];
}

// FIXME
void cudaGetPublicKeyStrawMan(unsigned long long *cpu_gpu_keys, unsigned long long **gpu_gpu_keys_addr){
    unsigned long long p, q, n, e, d;
    
    /* The BUG with RSA En/Decryption algorithm is that, we must group the data to be encrypted so that every block `m`'s size is LESS THAN `n`! */
    
    // This RSA pair is the same as the user's keys, which won't work properly (`n` encrypted by `(n,e)` would generate 0)
    // p = 74531;
    // q = 37019;
    // e = 0x10001;
    // d = 985968293;
    // n = p * q;
    
    // This RSA pair doesn't work :(
    // p = 61813;
    // q = 29347;
    // e = 0x10001;
    // d = 471467537;
    // n = p * q;

    // This RSA pair works :)
    p = 74831;
    q = 37619;
    e = 0x10001;
    d = 780225773;
    n = p * q;

    unsigned long long prime_host[3] = {d, n, e};

    cudaMalloc(&prime_pointer,3*sizeof(unsigned long long));

    cudaMemcpy(prime_pointer,&prime_host,sizeof(prime_host),cudaMemcpyHostToDevice);
    
    //Debug output
    unsigned long long test[3];
    cudaMemcpy(&test,prime_pointer,sizeof(test),cudaMemcpyDeviceToHost);
    printf("(debug)GPU's keys in host memory:%llu %llu %lld\n",prime_host[0],prime_host[1],prime_host[2]);
    printf("(debug)GPU's keys in device memory:%llu %llu %lld\n",test[0],test[1],test[2]);

    for(int i=0;i<2;i++) cpu_gpu_keys[i] = prime_host[i + 1]; // only copy (n, e) to host
    *gpu_gpu_keys_addr = prime_pointer;
    // printf("prime_pointer: %p\t*gpu_gpu_keys_addr: %p\n", prime_pointer, *gpu_gpu_keys_addr);
}
#ifndef _DEBUG

void cudaDecryptUserKeys(void* encrypted_user_keys, void *gpu_gpu_keys, void** gpu_user_keys_addr)
{
    cudaMalloc(&gpu_user_keys, 3*sizeof(unsigned long long)); // malloc for global address `gpu_user_keys`
    cudaMemcpy(gpu_user_keys, encrypted_user_keys, sizeof(unsigned long long) * 3, cudaMemcpyHostToDevice); // copy encrypted user keys to device
    decrypt_gpu(gpu_user_keys, 3 * sizeof(unsigned long long)/sizeof(int), gpu_gpu_keys); // decrypt user's keys
    *gpu_user_keys_addr = gpu_user_keys;

    unsigned long long test[3];
    cudaMemcpy(test, gpu_user_keys, sizeof(unsigned long long) * 3, cudaMemcpyDeviceToHost); // copy to host to debug
    printf("Decrypted user's keys on GPU: test[0] = %u, test[1] = %u, test[2] = %u\n", test[0], test[1], test[2]);
}
#else
int main(void){
    unsigned long long cpu_gpu_keys[2];
    unsigned long long *gpu_gpu_keys;
    cudaGetPublicKey(cpu_gpu_keys,&gpu_gpu_keys);
    printf("\nmain_real:");
    for(int i=0;i<2;i++)printf("%llu ",cpu_gpu_keys[i]);
    // cudaGetPublicKeyStrawMan(cpu_gpu_keys,&gpu_gpu_keys);
    // for(int i=0;i<2;i++)printf("%llu ",cpu_gpu_keys[i]);
    
    return 0;
}
#endif
