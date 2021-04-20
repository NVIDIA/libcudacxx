# Dockerfile for libcudacxx:host_ppc64le_centos_8.3.2011__target_ppc64le_centos_8.3.2011__gcc_8_cxx_14

FROM libcudacxx_base:host_ppc64le_centos_8.3.2011__target_ppc64le_centos_8.3.2011__gcc_8_cxx_14

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

###############################################################################
# BUILD: The following is invoked when the CUDA driver is attached.

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

ADD nvidia-modprobe /usr/bin/
# Restore setuid.
RUN chmod +s /usr/bin/nvidia-modprobe

ADD libcuda.so* /usr/lib/ppc64le-linux-gnu/
ADD libnvidia-ptxjitcompiler.so* /usr/lib/ppc64le-linux-gnu/

###############################################################################
# CMD: The following is invoked when the image is run.

RUN useradd -ms /bin/bash libcudacxx && chown -R libcudacxx:libcudacxx /sw
USER libcudacxx

WORKDIR /sw/gpgpu/libcudacxx

CMD /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash

