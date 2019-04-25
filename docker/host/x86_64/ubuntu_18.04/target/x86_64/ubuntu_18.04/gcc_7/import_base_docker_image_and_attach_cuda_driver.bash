#! /bin/bash

HOSTARCH=x86_64
HOSTOSKIND=ubuntu
HOSTOSVERSION=18.04
HOSTOS=${HOSTOSKIND}_${HOSTOSVERSION}
TARGETARCH=x86_64
TARGETOSKIND=ubuntu
TARGETOSVERSION=18.04
TARGETOS=${TARGETOSKIND}_${TARGETOSVERSION}
COMPILERKIND=gcc
COMPILERVERSION=7
COMPILER=${COMPILERKIND}_${COMPILERVERSION}

SCRIPTPATH=$(cd "$(dirname "$0")"; pwd -P)

SWPATH=$(realpath ${SCRIPTPATH}/../../../../../../../../../..)

if [ "" == "$1" ]
then
  IMAGE=libcudacxx_base__host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}.tar.bz2
else
  IMAGE="$1"
fi

bunzip2 ${IMAGE}

docker load --input ${IMAGE%.tar.bz2}.tar

${SCRIPTPATH}/attach_cuda_driver_to_docker_image.bash

docker image rm libcudacxx_base:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}

rm ${IMAGE%.tar.bz2}.tar

