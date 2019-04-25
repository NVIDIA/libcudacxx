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

TMPPATH=$(mktemp -d --suffix=-libcudacxx__host_${hostarch}_${hostos}__target_${targetarch}_${targetos}__${compiler})

LIBCUDA=$(ldconfig -p | grep libcuda.so | tr ' ' '\n' | grep / | tr '\n' ' ' | sed 's/ *$//')
LIBNVIDIAFATBINARYLOADER=$(ldconfig -p | grep libnvidia-fatbinaryloader.so | tr ' ' '\n' | grep / | tr '\n' ' ' | sed 's/ *$//')
LIBNVIDIAPTXJITCOMPILER=$(ldconfig -p | grep libnvidia-ptxjitcompiler.so | tr ' ' '\n' | grep / | tr '\n' ' ' | sed 's/ *$//')

echo "CUDA driver libraries found:"

for library in ${LIBCUDA} ${LIBNVIDIAFATBINARYLOADER} ${LIBNVIDIAPTXJITCOMPILER}
do
  echo "  ${library}"
done

cp ${LIBCUDA} ${LIBNVIDIAFATBINARYLOADER} ${LIBNVIDIAPTXJITCOMPILER} ${TMPPATH}

docker -D build \
  -t libcudacxx:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER} \
  -f ${SWPATH}/gpgpu/libcudacxx/docker/host/${HOSTARCH}/${HOSTOS}/target/${TARGETARCH}/${TARGETOS}/${COMPILER}/Dockerfile \
  ${TMPPATH}

rm ${TMPPATH}/*
rmdir ${TMPPATH}

