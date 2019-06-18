#! /bin/bash

SCRIPTPATH=$(cd $(dirname ${0}); pwd -P)

source ${SCRIPTPATH}/variant_configuration.bash

SWPATH=$(realpath ${SCRIPTPATH}/../../../../../../../../../..)

# Arguments are a list of SM architectures to target; if there are no arguments,
# all known SM architectures are targeted.

TARGETS=""

if [ ! -z "$@" ]
then
  TARGETS="-eLIBCUDACXX_COMPUTE_ARCHS=\"$@\""
fi

docker run $TARGETS --privileged libcudacxx:host_${HOSTARCH}_${HOSTOS}__target_${TARGETARCH}_${TARGETOS}__${COMPILER}

