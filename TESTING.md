# Testing `libcu++`

This document will describe how to build and run the `libcu++` test suite.

## *nix Systems

The procedure is demonstrated for NVCC + GCC in C++11 mode on a Debian-like
Linux systems; the same basic steps are required on all other platforms.

### Step 0: Install Prerequisites

In a Bash shell:

```
# Install LLVM (needed for LLVM's CMake modules)
apt-get -y install llvm

# Install CMake
apt-get -y install cmake 

# Install the LLVM Integrated Tester (`lit`)
apt-get -y install python-pip 
pip install lit
```

### Step 1: Generate the Build Files

In a Bash shell:

```
cd libcudacxx # This should be //sw/gpgpu/libcudacxx or the repo root.
mkdir -p build
cd build
cmake .. \
  -DLLVM_EXTERNAL_LIT=$(which lit) \
  -DLLVM_CONFIG_PATH=$(which llvm-config) \
  -DCMAKE_CXX_COMPILER=nvcc \
  -DLIBCXX_NVCC_HOST_COMPILER=g++ \
  -DLIBCXX_TEST_STANDARD_VER=c++11
```

### Step 2: Build and Run the Tests

In a Bash shell:

```
cd libcudacxx/build # `libcudacxx` should be //sw/gpgpu/libcudacxx or the Git repo root.
../utils/nvidia/linux/perform_tests.bash
```

## Windows

### Step 0: Install Prerequisites

Install [Git for Windows](https://git-scm.com/download/win):

Checkout [the LLVM Git mono repo](https://github.com/llvm/llvm-project) using a Git Bash shell:

```
git clone https://github.com/llvm/llvm-project.git /path/to/llvm
```

[Install Python](https://www.python.org/downloads/windows).

Download [the get-pip.py bootstrap script](https://bootstrap.pypa.io/get-pip.py) and run it.

Install the LLVM Integrated Tester (`lit`) using a Visual Studio command prompt:

```
pip install lit
```

### Step 1: Generate the Build Files

In a Visual Studio command prompt:

```
cd libcudacxx # This should be //sw/gpgpu/libcudacxx or the repo root.
mkdir build
cd build
cmake .. ^
  -G "Ninja" ^
  -DCMAKE_CXX_COMPILER_FORCED=ON ^
  -DCMAKE_C_COMPILER_FORCED=ON ^
  -DLLVM_EXTERNAL_LIT=lit ^
  -DLLVM_PATH="/path/to/llvm" ^
  -DCMAKE_CXX_COMPILER=nvcc ^
  -DLIBCXX_NVCC_HOST_COMPILER=cl
```

### Step 2: 

In a Visual Studio command prompt:

```
cd libcudacxx\build # `libcudacxx` should be //sw/gpgpu/libcudacxx or the Git repo root.
set LIBCXX_SITE_CONFIG=libcxx\test\lit.site.cfg
lit ..\test -vv -a
```

