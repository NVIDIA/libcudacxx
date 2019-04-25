# Dockerfile for libcudacxx_base:host_x86_64_ubuntu_18.04__target_x86_64_ubuntu_18.04__gcc_7

FROM ubuntu:18.04

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

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

# Copy NVCC and the CUDA runtime from the Perforce tree.
ADD bin /sw/gpgpu/bin

# Copy the core CUDA headers from the Perforce tree.
ADD cuda/import/*.h* /sw/gpgpu/cuda/import/
ADD cuda/common/*.h* /sw/gpgpu/cuda/common/
ADD cuda/tools/cudart/*.h* /sw/gpgpu/cuda/tools/cudart/
ADD cuda/tools/cudart/nvfunctional /sw/gpgpu/cuda/tools/cudart/
ADD cuda/tools/cnprt/*.h* /sw/gpgpu/cuda/tools/cnprt/
ADD cuda/tools/cooperative_groups/*.h* /sw/gpgpu/cuda/tools/cooperative_groups/
ADD cuda/tools/cudart/cudart_etbl/*.h* /sw/gpgpu/cuda/tools/cudart/cudart_etbl/
ADD opencl/import/cl_rel/CL/*.h* /sw/gpgpu/opencl/import/cl_rel/CL/

# Copy libcu++ sources from the Perforce tree.
ADD libcudacxx /sw/gpgpu/libcudacxx

# Configure libc++ tests.
RUN cd /sw/gpgpu/libcudacxx/libcxx/build\
 && cmake ..\
      -DLIBCXX_INCLUDE_TESTS=ON\
      -DLIBCXX_INCLUDE_BENCHMARKS=OFF\
      -DLIBCXX_CXX_ABI=libsupc++\
      -DLLVM_EXTERNAL_LIT=$(which lit)\
      -DLLVM_CONFIG_PATH=$(which llvm-config-6.0)\
      -DCMAKE_C_COMPILER=gcc-7\
      -DCMAKE_CXX_COMPILER=g++-7

# Configure libcu++ tests.
RUN cd /sw/gpgpu/libcudacxx/build\
 && cmake ..\
      -DLIBCXX_INCLUDE_TESTS=ON\
      -DLIBCXX_INCLUDE_BENCHMARKS=OFF\
      -DLIBCXX_CXX_ABI=libsupc++\
      -DLLVM_EXTERNAL_LIT=$(which lit)\
      -DLLVM_CONFIG_PATH=$(which llvm-config-6.0)\
      -DCMAKE_C_COMPILER=/sw/gpgpu/bin/x86_64_Linux_release/nvcc\
      -DCMAKE_CXX_COMPILER=/sw/gpgpu/bin/x86_64_Linux_release/nvcc\
      -DLIBCXX_HOST_COMPILER=g++-7

