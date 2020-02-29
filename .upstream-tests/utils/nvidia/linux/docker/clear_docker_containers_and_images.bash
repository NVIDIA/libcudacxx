#! /bin/bash

docker container rm $(docker container ls -a | awk '{ print $1 }' | grep -v CONTAINER) 2>&1 /dev/null
docker image rm $(docker image ls -a | awk '{ print $3 }'  | grep -v IMAGE) 2>&1 /dev/null

exit 0

