#! /bin/bash

SCRIPTPATH=$(cd "$(dirname "$0")"; pwd -P)

SWPATH=$(realpath ${SCRIPTPATH}/../../../../../../../../../..)

${SCRIPTPATH}/import_base_docker_image_and_attach_cuda_driver.bash

${SCRIPTPATH}/run_docker_image.bash

