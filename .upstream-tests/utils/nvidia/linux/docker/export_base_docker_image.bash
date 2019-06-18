#! /bin/bash

SCRIPTPATH=$(cd $(dirname ${0}); pwd -P)

source ${SCRIPTPATH}/variant_configuration.bash

SWPATH=$(realpath ${SCRIPTPATH}/../../../../../../../../../..)

# Arguments are a list of SM architectures to target; if there are no arguments,
# all known SM architectures are targeted.

if [ ! -z "$@" ]
then
  export LIBCUDACXX_COMPUTE_ARCHS="$@"
fi

${SCRIPTPATH}/build_base_docker_image.bash

if [ "$?" != "0" ]; then exit 1; fi

docker save \
  --output libcudacxx_base__host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}.tar \
  libcudacxx_base:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER} 

if [ "$?" != "0" ]; then exit 1; fi

bzip2 --force libcudacxx_base__host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}.tar

docker image rm libcudacxx_base:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}
docker image rm ${HOSTOSKIND}:${HOSTOSVERSION}

