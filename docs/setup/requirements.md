# Requirements

All requirements are applicable to the `main` branch on GitHub.
For details on specific releases, please see the [changelog].

## Usage Requirements

To use the NVIDIA C++ Standard Library, you must follow meet the following
  requirements.

### System Software

The NVIDIA C++ Standard Library requires either the [NVIDIA HPC SDK] or the
  [CUDA Toolkit].

libcu++ was first released in NVHPC 20.3 and CUDA 10.2.
Some features are only available in newer releases.
Please see the [API section] and the [release section] to find which features
  require newer releases.

Releases of libcu++ are only tested against the latest releases of NVHPC and
  CUDA.
It may be possible to use a newer version of libcu++ with an older NVHPC or
  CUDA installation by using a libcu++ release from GitHub, but please be aware
  this is not officially supported.

### C++ Dialects

The NVIDIA C++ Standard Library supports the following C++ dialects:

- C++11
- C++14
- C++17

A number of features have been backported to earlier standards.
Please see the [API section] for more details.

### NVCC Host Compilers

When used with NVCC, the NVIDIA C++ Standard Library supports the following host
  compilers:

- MSVC 2017 and 2019.
- GCC 5, 6, 7, 8, 9, and 10.
- Clang 7, 8, 9, and 10.
- ICPC 19.
- NVHPC 20.7 and 20.9

### Device Architectures

The NVIDIA C++ Standard Library fully supports the following NVIDIA device
  architectures:

- Volta: SM 70 and 72.
- Turing: SM 75.
- Ampere: SM 80.

The following NVIDIA device architectures are partially supported:

- Maxwell: SM 50, 52 and 53.
    - Synchronization facilities are supported.
- Pascall: SM 60, 61 and 62.
    - Blocking synchronization facilities (e.g. most of the synchronization
          library) are not supported.
        Please see the [synchronization library section] for details.

### Host Architectures

The NVIDIA C++ Standard Library supports the following host architectures:

- aarch64.
- x86-64.
- ppc64le.

### Host Operating Systems

The NVIDIA C++ Standard Library supports the following host operating systems:

- Linux.
- Windows.
- Android.
- QNX.

## Build and Test Requirements

To build and test libcu++ yourself, you will need the following in addition to
  the usage requirements:

- [CMake].
- [LLVM].
  - You do not have to build LLVM; we only need its CMake modules.
- [lit], the LLVM Integrated Tester.
  - We recommend installing lit using Python's pip package manager.


[API section]: ./api.md
[synchronization library section]: https://nvidia.github.io/libcudacxx/api/synchronization_library.html
[changelog]: ./releases/changelog.md

[NVIDIA HPC SDK]: https://developer.nvidia.com/hpc-sdk
[CUDA Toolkit]: https://developer.nvidia.com/cuda-toolkit

[CMake]: https://cmake.org
[LLVM]: https://github.com/llvm
[lit]: https://pypi.org/project/lit/
