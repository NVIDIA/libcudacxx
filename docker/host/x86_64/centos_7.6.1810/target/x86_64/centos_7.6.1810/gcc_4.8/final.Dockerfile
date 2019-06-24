# Dockerfile for libcudacxx:host_x86_64_centos_7.6__target_x86_64_centos_7.6__gcc_4.8

FROM libcudacxx_base:host_x86_64_centos_7.6__target_x86_64_centos_7.6__gcc_4.8

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

###############################################################################
# BUILD: The following is invoked when the CUDA driver is attached.

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

ADD libcuda.so* /usr/lib64/
ADD libnvidia-fatbinaryloader.so* /usr/lib64/
ADD libnvidia-ptxjitcompiler.so* /usr/lib64/

###############################################################################
# CMD: The following is invoked when the image is run.

WORKDIR /sw/gpgpu/libcudacxx

ENV LIBCUDACXX_COMPUTE_ARCHS="30 32 35 50 52 53 60 61 62 70 72 75"

CMD /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash

