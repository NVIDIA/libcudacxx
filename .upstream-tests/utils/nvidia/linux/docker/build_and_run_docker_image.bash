#! /bin/bash

SCRIPTPATH=$(cd $(dirname ${0}); pwd -P)

source ${SCRIPTPATH}/variant_configuration.bash

SWPATH=$(realpath ${SCRIPTPATH}/../../../../../../../../../..)

LIBCUDACXX_SKIP_BASE_TESTS_BUILD=1 ${SCRIPTPATH}/build_base_docker_image.bash

${SCRIPTPATH}/attach_cuda_driver_to_docker_image.bash

if [ "$?" != "0" ]; then exit 1; fi

${SCRIPTPATH}/run_docker_image.bash $@

