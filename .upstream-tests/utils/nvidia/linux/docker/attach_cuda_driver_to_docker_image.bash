#! /bin/bash

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)
source ${SCRIPT_PATH}/configuration.bash

TMP_PATH=$(mktemp -d --suffix=-${FINAL_NAME})
LIBCUDA=$(ldconfig -p | grep libcuda.so | tr ' ' '\n' | grep / | tr '\n' ' ' | sed 's/ *$//')
LIBNVIDIAFATBINARYLOADER=$(ldconfig -p | grep libnvidia-fatbinaryloader.so | tr ' ' '\n' | grep / | tr '\n' ' ' | sed 's/ *$//')
LIBNVIDIAPTXJITCOMPILER=$(ldconfig -p | grep libnvidia-ptxjitcompiler.so | tr ' ' '\n' | grep / | tr '\n' ' ' | sed 's/ *$//')

echo "CUDA driver libraries found:"

for library in ${LIBCUDA} ${LIBNVIDIAFATBINARYLOADER} ${LIBNVIDIAPTXJITCOMPILER}
do
  echo "  ${library}"
done

cp ${LIBCUDA} ${LIBNVIDIAFATBINARYLOADER} ${LIBNVIDIAPTXJITCOMPILER} ${TMP_PATH}
if [ "${?}" != "0" ]; then exit 1; fi

docker -D build -t ${FINAL_IMAGE} -f ${FINAL_DOCKERFILE} ${TMP_PATH}
if [ "${?}" != "0" ]; then exit 1; fi

rm ${TMP_PATH}/*
rmdir ${TMP_PATH}

