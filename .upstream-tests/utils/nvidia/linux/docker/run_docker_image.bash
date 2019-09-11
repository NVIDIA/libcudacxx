#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

# Arguments are a list of SM architectures to target; if there are no arguments,
# all known SM architectures are targeted.
COMPUTE_ARCHS_FLAG=""
if [ ! -z "${@}" ]
then
  COMPUTE_ARCHS_FLAG="-eLIBCUDACXX_COMPUTE_ARCHS=\"${@}\""
fi

docker run -t ${COMPUTE_ARCHS_FLAG} --privileged ${FINAL_IMAGE}
if [ "${?}" != "0" ]; then exit 1; fi

