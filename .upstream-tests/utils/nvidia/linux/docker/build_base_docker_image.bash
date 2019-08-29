#! /bin/bash

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

# Arguments are a list of SM architectures to target; if there are no arguments,
# all known SM architectures are targeted.

# Pull the OS image from Docker Hub; try a few times in case our connection is
# slow.
docker pull ${OS_IMAGE} \
|| docker pull ${OS_IMAGE} \
|| docker pull ${OS_IMAGE} \
|| docker pull ${OS_IMAGE}

# Copy the .dockerignore file from //sw/gpgpu/libcudacxx to //sw/gpgpu.
cp ${SW_PATH}/gpgpu/libcudacxx/docker/.dockerignore ${SW_PATH}/gpgpu

LIBCUDACXX_COMPUTE_ARCHS="${@}" docker -D build \
  --build-arg LIBCUDACXX_SKIP_BASE_TESTS_BUILD \
  --build-arg LIBCUDACXX_COMPUTE_ARCHS \
  -t ${BASE_IMAGE} \
  -f ${BASE_DOCKERFILE} \
  ${SW_PATH}/gpgpu
if [ "${?}" != "0" ]; then exit 1; fi

# Create a temporary container so we can extract the log files.
TMP_CONTAINER=$(docker create ${BASE_IMAGE})

docker cp ${TMP_CONTAINER}:/sw/gpgpu/libcudacxx/libcxx/build/cmake_libcxx.log .
docker cp ${TMP_CONTAINER}:/sw/gpgpu/libcudacxx/build/cmake_libcudacxx.log .
docker cp ${TMP_CONTAINER}:/sw/gpgpu/libcudacxx/build/lit.log .
docker cp ${TMP_CONTAINER}:/sw/gpgpu/libcudacxx/build/lit_sm6x_plus.log .

docker container rm ${TMP_CONTAINER} > /dev/null

# Remove the .dockerignore from //sw/gpgpu.
rm -f ${SW_PATH}/gpgpu/.dockerignore

