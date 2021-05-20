#ifndef ENCHECAP_HOST_H__
#define ENCHECAP_HOST_H__

#include "Enchecap_common.h"

/* SGX */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pwd.h>
#include "sgx_urts.h"
#include "Enclave_u.h"
#include "ErrorSGX.h"

# define MAX_PATH FILENAME_MAX
# define TOKEN_FILENAME   "enclave.token"
# define ENCLAVE_FILENAME "enclave.signed.so"

////////////////////////////////////////////////////////////////////////////////
//! Initialize Enchecap environment
//! @param eid        Must be `global_eid` defined in App.cpp
//! @param ecpreg     Gloabl Enchecap context
////////////////////////////////////////////////////////////////////////////////
int initEnchecap(unsigned long &eid, ECPreg *ecpreg);
 ////////////////////////////////////////////////////////////////////////////////
 //! init a transfer of GPU RSA public key
 //! pk is an array containing 3 64-bit unsigned int d, n and e accordingly
 //! whereas encrypt function being f(m)= m^d mod n, encrypt function g(m)=m^e mod n
 //! user should apply for mem space beforehand
 //! @param des        destination of the key pair copy
 ////////////////////////////////////////////////////////////////////////////////
void cudaGetPublicKey(unsigned long long *des);
void cudaGetPublicKeyStrawMan(unsigned long long *cpu_gpu_keys, unsigned long long **gpu_gpu_keys_addr); // FIXME
void cudaGetSGXKey(void* shared_sgx_pub_key_host, void** shared_sgx_pub_key_device_addr);
#endif