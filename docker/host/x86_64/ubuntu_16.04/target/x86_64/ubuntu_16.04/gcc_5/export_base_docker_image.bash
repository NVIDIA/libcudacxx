#! /bin/bash

HOSTARCH=x86_64
HOSTOSKIND=ubuntu
HOSTOSVERSION=16.04
HOSTOS=${HOSTOSKIND}_${HOSTOSVERSION}
TARGETARCH=x86_64
TARGETOSKIND=ubuntu
TARGETOSVERSION=16.04
TARGETOS=${TARGETOSKIND}_${TARGETOSVERSION}
COMPILERKIND=gcc
COMPILERVERSION=5
COMPILER=${COMPILERKIND}_${COMPILERVERSION}

SCRIPTPATH=$(cd "$(dirname "$0")"; pwd -P)

SWPATH=$(realpath ${SCRIPTPATH}/../../../../../../../../../..)

${SCRIPTPATH}/build_base_docker_image.bash

docker save \
  --output libcudacxx_base__host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}.tar \
  libcudacxx_base:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER} 

bzip2 libcudacxx_base__host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}.tar

docker image rm libcudacxx_base:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}
docker image rm ${HOSTOSKIND}:${HOSTOSVERSION}

