# `/App` File Structure

## Enchecap Library
* `Enchecap_commom.h`: including definition of `struct ECPreg`
* `Enchecap_host.h/cpp`:
    
    ```Cpp
    ////////////////////////////////////////////////////////////////////////////////
    //! Initialize Enchecap environment
    //! @param eid        Must be `global_eid` defined in App.cpp
    //! @param ecpreg     Gloabl Enchecap context
    ////////////////////////////////////////////////////////////////////////////////
    int initEnchecap(unsigned long &eid, ECPreg ecpreg);
    ```
* `Enchecap_device.h/cpp`:
    
    ```Cpp
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
    ```
* `ErrorSGX.h/cpp`: including SGX error info definition
* `EnclaveWrapper.h/cpp`: a necessary wrapper for encryption/decryption ecalls
* `DeviceRSA.h/cpp`: a necessary wrapper for encryption/decryption GPU kernel functions

## Other File Structure

* `App.h/cpp`: application entrance
* `matrixMulCUBLAS.h/cpp`: CUBLAS example code
* `DeviceApp.h/cpp`: a tiny demo calling GPU