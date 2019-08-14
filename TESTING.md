# Testing `libcu++`

This document will describe how to build and run the `libcu++` tests.

The procedure is demonstrated for NVCC + GCC in C++11 mode on a Debian-like
Linux systems; the same basic steps are required on all other platforms.

## Step 0: Install Prerequisites.

```
# Install LLVM (needed for LLVM's CMake modules)
apt-get -y install llvm

# Install CMake
apt-get -y install cmake 

# Install the LLVM Integrated Tester (`lit`)
apt-get -y install python-pip 
pip install lit
```

## Step 1: Generate the Build Files

```
cd libcudacxx # This should be the root of the `libcudacxx` repository
mkdir -p build
cd build
cmake .. \
  -DLIBCXX_STANDARD_VER=c++11 \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DLLVM_CONFIG_PATH=$(which llvm-config) \
  -DCMAKE_CXX_COMPILER=nvcc \
  -DLIBCXX_NVCC_HOST_COMPILER=g++
```

## Step 2: Build and Run the Tests

```
../utils/nvidia/linux/perform_tests.bash
```

