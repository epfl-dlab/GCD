#!/bin/sh

project="${1:-gcd}"  # Use "llama" if no argument is provided
tag="${2:-latest}"     # Use "latest" if no tag argument is provided
user="${3:-geng}"

docker build -f docker/Dockerfile  --build-arg USER_NAME=${user} --build-arg PROJECT_NAME=${project} -t ic-registry.epfl.ch/dlab/"${user}"/"${project}":"${tag}" --secret id=dot_env,src=docker/.env .

#docker push ic-registry.epfl.ch/dlab/"${user}"/"${project}":"${tag}"