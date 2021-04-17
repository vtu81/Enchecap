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
int initEnchecap(unsigned long &eid, ECPreg ecpreg);

#endif