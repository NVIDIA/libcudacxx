# Dockerfile for libcudacxx:host_x86_64_ubuntu_18.04__target_x86_64_ubuntu_18.04__gcc_7

###############################################################################
# BUILD: The following is invoked when the CUDA driver is attached.

FROM libcudacxx_base:host_x86_64_ubuntu_18.04__target_x86_64_ubuntu_18.04__gcc_7

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

###############################################################################
# BUILD: The following is invoked when the image is built.

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

ADD libcuda.so* /usr/lib/x86_64-linux-gnu/ 
ADD libnvidia-fatbinaryloader.so* /usr/lib/x86_64-linux-gnu/ 
ADD libnvidia-ptxjitcompiler.so* /usr/lib/x86_64-linux-gnu/ 

###############################################################################
# RUN: The following is invoked when the image is run.

WORKDIR /sw/gpgpu/libcudacxx

ENV LIBCXX_SITE_CONFIG /sw/gpgpu/libcudacxx/build/libcxx/test/lit.site.cfg

CMD LIBCXX_SITE_CONFIG=/sw/gpgpu/libcudacxx/libcxx/build/libcxx/test/lit.site.cfg lit -vv -a libcxx/test\
 && LIBCXX_SITE_CONFIG=/sw/gpgpu/libcudacxx/build/libcxx/test/lit.site.cfg lit -vv -a test

