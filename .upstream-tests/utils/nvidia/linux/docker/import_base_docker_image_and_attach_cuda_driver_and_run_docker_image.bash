#! /bin/bash

SCRIPTPATH=$(cd $(dirname ${0}); pwd -P)

source ${SCRIPTPATH}/variant_configuration.bash

${SCRIPTPATH}/import_base_docker_image_and_attach_cuda_driver.bash

if [ "$?" != "0" ]; then exit 1; fi

${SCRIPTPATH}/run_docker_image.bash $@

