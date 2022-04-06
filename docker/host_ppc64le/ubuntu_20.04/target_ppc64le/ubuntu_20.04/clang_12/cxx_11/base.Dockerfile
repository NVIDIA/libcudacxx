# Dockerfile for libcudacxx_base:host_ppc64le_ubuntu_20.04__target_ppc64le_ubuntu_20.04__clang_12_cxx_11

FROM ubuntu:20.04

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

ARG LIBCUDACXX_SKIP_BASE_TESTS_BUILD
ARG LIBCUDACXX_COMPUTE_ARCHS
ARG COMPILER_CXX_DIALECT

ENV COMPILER_CXX_DIALECT=$COMPILER_CXX_DIALECT
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Pacific

###############################################################################
# BUILD: The following is invoked when the image is built.

SHELL ["/usr/bin/env", "bash", "-c"]

RUN apt-get -y update\
 && DEBIAN_FRONTEND="noninteractive" TZ="US/Pacific"\
 	apt-get -y install clang-12 python3 python3-pip curl libarchive13 libjsoncpp-dev libuv1 libssl-dev\
 && python3 -m pip install lit\
 && mkdir -p /sw/gpgpu/libcudacxx/build\
 && mkdir -p /sw/gpgpu/libcudacxx/libcxx/build

ADD https://github.com/Kitware/CMake/archive/refs/tags/v3.22.3.tar.gz /tmp/cmake.install.tar.gz
RUN tar xvf /tmp/cmake.install.tar.gz -C /tmp/
RUN cd /tmp/CMake-3.22.3 && ./configure && make -j8 && make -j8 install

# For debugging.
#RUN apt-get -y install gdb strace vim

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

# Copy NVCC and the CUDA runtime from the source tree.
ADD bin /sw/gpgpu/bin

# Copy libcu++ sources from the source tree.
ADD cuda/tools/libcudacxxext /sw/gpgpu/cuda/tools/libcudacxxext
ADD libcudacxx /sw/gpgpu/libcudacxx

# List out everything in /sw before the build.
RUN echo "Contents of /sw:" && cd /sw/ && find

# Build libc++ and configure libc++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/libcxx/build\
 && cmake ../..\
 -DLIBCUDACXX_ENABLE_STATIC_LIBRARY=OFF\
 -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=OFF\
 -DLIBCUDACXX_ENABLE_LIBCXX_TESTS=ON\
 -DLIBCXX_TEST_STANDARD_VER=c++$COMPILER_CXX_DIALECT\
 -DCMAKE_CXX_COMPILER=clang++-12\
 -DCMAKE_CUDA_COMPILER=/sw/gpgpu/bin/ppc64le_Linux_release/nvcc\
 && make -j\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcxx.log

# Configure libcu++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/build\
 && cmake ..\
 -DCMAKE_CXX_COMPILER=clang++-12\
 -DCMAKE_CUDA_COMPILER=/sw/gpgpu/bin/ppc64le_Linux_release/nvcc\
 -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON\
 -DLIBCUDACXX_ENABLE_LIBCXX_TESTS=OFF\
 -DLIBCUDACXX_TEST_STANDARD_VER=c++$COMPILER_CXX_DIALECT\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcudacxx.log

# Build tests if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS=$LIBCUDACXX_COMPUTE_ARCHS\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_all.log

# Build tests for sm6x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="60 61 62"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm6x.log

# Build tests for sm7x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="70 72 75"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm7x.log

# Build tests for sm8x if requested.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx\
 && LIBCUDACXX_COMPUTE_ARCHS="80"\
 LIBCUDACXX_SKIP_BASE_TESTS_BUILD=$LIBCUDACXX_SKIP_BASE_TESTS_BUILD\
 /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash\
 --skip-tests-runs\
 --skip-libcxx-tests\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/build_lit_sm8x.log

# Package the logs up, because `docker container cp` doesn't support wildcards,
# and I want to limit the number of places we have to make changes when we ship
# new architectures.
RUN cd /sw/gpgpu/libcudacxx/build && tar -cvf logs.tar *.log

WORKDIR /sw/gpgpu/libcudacxx

