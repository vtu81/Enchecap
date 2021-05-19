# Enchecap

An (onbuilding) **enc**rypted (**enc**lave-based) **he**terogeneous **ca**lculation **p**rotocol based on Nvidia CUDA and Intel SGX, designed and implemented by [Tinghao Xie](http://vtu.life), [Haoyang Shi](https://github.com/Luke-Skycrawler), [Zihang Li](https://github.com/zjulzhhh).

Enchecap illustration:
![demo](./assets/demo.png)

Enchecap illustration (with **protected** and **trusted** regions):
![demo](./assets/demo_box.png)

---

To **build** the project, you'll need to install and configure:
* SGX SDK
* CUDA Toolkit
* CUDA Samples

, then set your `CUDA_PATH` and `INCLUDES` in Makefile, and make sure your SGX environment activated by

```bash
source /PATH_OF_SGXSDK/environment
```

(check SGX SDK official [site](https://01.org/intel-software-guard-extensions) for more details)

Then build with:

```bash
make # SGX hardware mode
```

```bash
make SGX_MODE=SIM  # SGX simulation mode
```

(check README_SGX.txt for more details)

> Your linux OS version might be limited by SGX SDK, check https://01.org/intel-software-guard-extensions for more details. We're using Ubuntu 18.04 x86_64, and cannot guarantee it work successfully on other platforms. We are also compiling with gcc version 7.5.0 and nvcc v11.1, which do not pose such strict limitations compared to Intel SGX.

---

To **run** the project, you'll need to install and configure correctly:
* SGX PSW
* SGX driver, if you build it in hardware mode and that your CPU & BIOS support SGX
* CUDA Driver (of course you must have an Nvidia GPU)

Run with:

```bash
./app
```

## TODO

### Phase I: Initialization
- [x] 1. Create an enclave
- [ ] 2. User broadcasts his/her public key to host & device
- [ ] 3. Enclave generates its own keys, then broadcasts its public key to user & device
- [ ] 4. GPU generates its own keys, then broadcasts its public key to host & user

### Phase II: Calculation
- [x] 6. En/Decrypt in enclave (Enclave's private key is only unsealed and visible in enclave only)
- [x] 7. En/Decrypt on GPU (GPU's private key is always in device memory)

### Future Work
- [ ] Test the performance
- [ ] The user's keys are now simply welded in the code, need FIX
- [ ] The GPU's keys are now simply welded in the code, need FIX
- [ ] The current RSA en/decrypt algorithm is yet extremely naive! (further works include regrouping, big number supports...)
- [ ] Remote attestation with Intel SGX
- [ ] Intergration with real industrial work based on CUDA
- [ ] Intergration with a trusted GPU (far from our reach now)
