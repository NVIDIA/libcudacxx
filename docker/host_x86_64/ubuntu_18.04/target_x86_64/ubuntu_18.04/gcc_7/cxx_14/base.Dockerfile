# Dockerfile for libcudacxx_base:host_x86_64_ubuntu_18.04__target_x86_64_ubuntu_18.04__gcc_7_cxx_14

FROM ubuntu:18.04

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>
SHELL ["/usr/bin/env", "bash", "-c"]

ARG LIBCUDACXX_SKIP_BASE_TESTS_BUILD
ARG LIBCUDACXX_COMPUTE_ARCHS

###############################################################################
# BUILD: The following is invoked when the image is built.

RUN apt-get -y update\
 && apt-get -y install g++-7 clang-6.0 python python-pip cmake\
 && pip install lit\
 && mkdir -p /sw/gpgpu/libcudacxx/build\
 && mkdir -p /sw/gpgpu/libcudacxx/libcxx/build

# For debugging.
#RUN apt-get -y install gdb strace vim

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

# Copy NVCC and the CUDA runtime from the source tree.
ADD bin /sw/gpgpu/bin

# Copy the core CUDA headers from the source tree.
ADD cuda/import/*.h* /sw/gpgpu/cuda/import/
ADD cuda/common/*.h* /sw/gpgpu/cuda/common/
ADD cuda/tools/cudart/*.h* /sw/gpgpu/cuda/tools/cudart/
ADD cuda/tools/cudart/nvfunctional /sw/gpgpu/cuda/tools/cudart/
ADD cuda/tools/cnprt/*.h* /sw/gpgpu/cuda/tools/cnprt/
ADD cuda/tools/cooperative_groups/*.h* /sw/gpgpu/cuda/tools/cooperative_groups/
ADD cuda/tools/cudart/cudart_etbl/*.h* /sw/gpgpu/cuda/tools/cudart/cudart_etbl/
ADD opencl/import/cl_rel/CL/*.h* /sw/gpgpu/opencl/import/cl_rel/CL/

# Copy libcu++ sources from the source tree.
ADD libcudacxx /sw/gpgpu/libcudacxx

# List out everything in /sw before the build.
RUN echo "Contents of /sw:" && cd /sw/ && find

# Build libc++ and configure libc++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/libcxx/build\
 && cmake ..\
 -DLIBCXX_INCLUDE_TESTS=ON\
 -DLIBCXX_INCLUDE_BENCHMARKS=OFF\
 -DLIBCXX_CXX_ABI=libsupc++\
 -DLIBCXX_TEST_STANDARD_VER=c++14\
 -DLIBCXX_ABI_UNSTABLE=ON\
 -DLLVM_CONFIG_PATH=$(which llvm-config-6.0)\
 -DCMAKE_C_COMPILER=gcc-7\
 -DCMAKE_CXX_COMPILER=g++-7\
 && make -j\
 2>&1 | tee /sw/gpgpu/libcudacxx/libcxx/build/cmake_libcxx.log

# Configure libcu++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/build\
 && cmake ..\
 -DLIBCXX_TEST_STANDARD_VER=c++14\
 -DLLVM_CONFIG_PATH=$(which llvm-config-6.0)\
 -DCMAKE_CXX_COMPILER=/sw/gpgpu/bin/x86_64_Linux_release/nvcc\
 -DLIBCXX_NVCC_HOST_COMPILER=g++-7\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcudacxx.log

# Build tests if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS=$LIBCUDACXX_COMPUTE_ARCHS\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/lit.log

# Build tests for sm6x and up if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="60 61 62 70 72 75"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/lit_sm6x_plus.log

WORKDIR /sw/gpgpu/libcudacxx

