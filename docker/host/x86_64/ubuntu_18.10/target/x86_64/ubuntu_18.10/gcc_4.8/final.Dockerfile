# Dockerfile for libcudacxx:host_x86_64_ubuntu_18.10__target_x86_64_ubuntu_18.10__gcc_4.8

FROM libcudacxx_base:host_x86_64_ubuntu_18.10__target_x86_64_ubuntu_18.10__gcc_4.8

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

###############################################################################
# BUILD: The following is invoked when the CUDA driver is attached.

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

ADD libcuda.so* /usr/lib/x86_64-linux-gnu/ 
ADD libnvidia-fatbinaryloader.so* /usr/lib/x86_64-linux-gnu/ 
ADD libnvidia-ptxjitcompiler.so* /usr/lib/x86_64-linux-gnu/ 

###############################################################################
# CMD: The following is invoked when the image is run.

WORKDIR /sw/gpgpu/libcudacxx

ENV LIBCUDACXX_COMPUTE_ARCHS="30 32 35 50 52 53 60 61 62 70 72 75"

CMD /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash

