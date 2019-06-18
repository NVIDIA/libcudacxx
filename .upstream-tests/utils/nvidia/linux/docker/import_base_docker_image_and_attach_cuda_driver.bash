#! /bin/bash

SCRIPTPATH=$(cd $(dirname ${0}); pwd -P)

source ${SCRIPTPATH}/variant_configuration.bash

SWPATH=$(realpath ${SCRIPTPATH}/../../../../../../../../../..)

if [ "" == "$1" ]
then
  IMAGE=libcudacxx_base__host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}.tar.bz2
else
  IMAGE="$1"
fi

bunzip2 ${IMAGE}

if [ "$?" != "0" ]; then exit 1; fi

docker load --input ${IMAGE%.tar.bz2}.tar

if [ "$?" != "0" ]; then exit 1; fi

${SCRIPTPATH}/attach_cuda_driver_to_docker_image.bash

if [ "$?" != "0" ]; then exit 1; fi

docker image rm libcudacxx_base:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}

rm ${IMAGE%.tar.bz2}.tar

