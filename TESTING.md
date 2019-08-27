# Testing `libcu++`

This document will describe how to build and run the `libcu++` test suite.

## *nix Systems, Native Build/Test

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
export LIBCUDACXX_ROOT=/path/to/libcudacxx # Should be //sw/gpgpu/libcudacxx or the Git repo root.

cd ${LIBCUDACXX_ROOT}
mkdir -p build
cd build
cmake .. \
  -DLLVM_CONFIG_PATH=$(which llvm-config) \
  -DCMAKE_CXX_COMPILER=nvcc \
  -DLIBCXX_NVCC_HOST_COMPILER=g++ \
  -DLIBCXX_TEST_STANDARD_VER=c++11
```

### Step 2: Build and Run the Tests

In a Bash shell:

```
export LIBCUDACXX_ROOT=/path/to/libcudacxx # Should be //sw/gpgpu/libcudacxx or the Git repo root.

cd ${LIBCUDACXX_ROOT}/build # `libcudacxx` should be //sw/gpgpu/libcudacxx or the Git repo root.
../utils/nvidia/linux/perform_tests.bash --skip-libcxx-tests
```

## *nix Systems, Cross Build/Test

The procedure is demonstrated for NVCC + GCC cross compiler in C++14 mode on a
Debian-like Linux systems targeting an aarch64 L4T system; the same basic steps
are required on all other platforms.

### Step 0:

Follow Step 0 for *nix native builds/tests.

### Step 1:

In a Bash shell:

```
export HOST=executor.nvidia.com
export USERNAME=ubuntu
export LIBCUDACXX_ROOT=/path/to/libcudacxx # Should be //sw/gpgpu/libcudacxx or the Git repo root.

cd ${LIBCUDACXX_ROOT}
mkdir -p build
cd build
cmake .. \
  -DLLVM_CONFIG_PATH=$(which llvm-config) \
  -DCMAKE_CXX_COMPILER=nvcc \
  -DLIBCXX_NVCC_HOST_COMPILER=aarch64-linux-gnu-g++ \
  -DLIBCXX_TEST_STANDARD_VER=c++14 \
  -DLIBCXX_EXECUTOR="SSHExecutor(host='${HOST}', username='${USERNAME}')"
```

Ensure that you can SSH to the target system from the host system without
inputing a password (e.g. use SSH keys).

### Step 2:

Follow Step 2 for *nix native builds/tests.

## Windows, Native Build/Test

### Step 0: Install Prerequisites

Install [Git for Windows](https://git-scm.com/download/win):

Checkout [the LLVM Git mono repo](https://github.com/llvm/llvm-project) using a
Git Bash shell:

```
export LLVM_ROOT=/path/to/llvm

git clone https://github.com/llvm/llvm-project.git ${LLVM_ROOT}
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
set LLVM_ROOT=\path\to\llvm
set LIBCUDACXX_ROOT=\path\to\libcudacxx # Should be //sw/gpgpu/libcudacxx or the Git repo root.

cd %LIBCUDACXX_ROOT% 
mkdir build
cd build
cmake .. ^
  -G "Ninja" ^
  -DLLVM_PATH=%LLVM_ROOT%\llvm ^
  -DCMAKE_CXX_COMPILER=nvcc ^
  -DLIBCXX_NVCC_HOST_COMPILER=cl ^
  -DCMAKE_CXX_COMPILER_FORCED=ON ^
  -DCMAKE_C_COMPILER_FORCED=ON
```

### Step 2: 

In a Visual Studio command prompt:

```
set LIBCUDACXX_ROOT=\path\to\libcudacxx # Should be //sw/gpgpu/libcudacxx or the Git repo root.
set SM_ARCH=70

cd %LIBCUDACXX_ROOT%\build
set LIBCXX_SITE_CONFIG=libcxx\test\lit.site.cfg
lit ..\test -Dcompute_archs=%SM_ARCH% -vv -a 
```

