#! /bin/bash

set -e

docker container rm $(docker container ls -a | awk '{ print $1 }' | grep -v CONTAINER)
docker image rm $(docker image ls -a | awk '{ print $3 }'  | grep -v IMAGE)

