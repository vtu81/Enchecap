#ifndef ENCHECAP_DEVICE_H__
#define ENCHECAP_DEVICE_H__

#include "Enchecap_common.h"

// RSA
#include "DeviceRSA.h"
#include "EnclaveWrapper.h"

////////////////////////////////////////////////////////////////////////////////
//! Secure CUDA memory copy
//! @param ecpreg     Global Enchecap context
//! @param dst        Destination memory address
//! @param src        Source memory address
//! @param count      Size in bytes to copy
//! @param kind       Type of transfer
//! @param encrypt_s  Whether to encrypt before copying data from the source or not (1 or 0)
//! @param decrypt_d  Whether to decrypt after data's arrival at the destination or not (1 or 0)
////////////////////////////////////////////////////////////////////////////////
cudaError_t secureCudaMemcpy(ECPreg ecpreg, void *dst, void *src, size_t count, enum cudaMemcpyKind kind, int encrypt_s, int decrypt_d);

#endif