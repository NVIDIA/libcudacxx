#! /bin/bash

SCRIPTPATH=$(cd $(dirname ${0}); pwd -P)

source ${SCRIPTPATH}/variant_configuration.bash

SWPATH=$(realpath ${SCRIPTPATH}/../../../../../../../../../..)

# Copy the .dockerignore file from //sw/gpgpu/libcudacxx to //sw/gpgpu.
cp ${SWPATH}/gpgpu/libcudacxx/docker/.dockerignore ${SWPATH}/gpgpu

docker -D build \
  --build-arg LIBCUDACXX_SKIP_BASE_TESTS_BUILD \
  -t libcudacxx_base:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER} \
  -f ${SWPATH}/gpgpu/libcudacxx/docker/host/${HOSTARCH}/${HOSTOS}/target/${TARGETARCH}/${TARGETOS}/${COMPILER}/base.Dockerfile \
  ${SWPATH}/gpgpu

if [ "$?" != "0" ]; then exit 1; fi

# Remove the .dockerignore from //sw/gpgpu.
rm ${SWPATH}/gpgpu/.dockerignore

