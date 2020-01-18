#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

NVIDIAMODPROBE=$(which nvidia-modprobe)

# Arguments are a list of SM architectures to target; if there are no arguments,
# all known SM architectures are targeted.
COMPUTE_ARCHS_FLAG=""
if [ ! -z "${@}" ]
then
  COMPUTE_ARCHS_FLAG="-eLIBCUDACXX_COMPUTE_ARCHS=\"${@}\""
fi

# Ensure nvidia-uvm is loaded.
${NVIDIAMODPROBE} -u

docker run -t ${COMPUTE_ARCHS_FLAG} --privileged ${FINAL_IMAGE} 2>&1 \
  | while read l; do \
      echo "${LIBCUDACXX_DOCKER_OUTPUT_PREFIX}$(date --rfc-3339=seconds)| $l"; \
    done
if [ "${PIPESTATUS[0]}" != "0" ]; then exit 1; fi

