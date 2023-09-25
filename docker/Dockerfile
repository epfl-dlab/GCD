# Choose a docker template, with ARG you can set the base image from the command line
# for example: docker build --build-arg BASE_IMAGE=xxx --build-arg USERNAME=xxx --build-arg PROJECT_NAME=xxx .
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.02-py3
FROM $BASE_IMAGE
ARG USER_NAME
ARG PROJECT_NAME

# install sudo (necessary for adding a new user, see below)
RUN apt update && \
    apt -y install sudo

# Setup your user profile with the right group permission to access NFS folder
# For the command that gives the ids and names you would need checkout the .env file
RUN --mount=type=secret,id=dot_env source /run/secrets/dot_env && \
     groupadd -g ${GROUP_ID1} ${GROUP_NAME1} && \
     groupadd -g ${GROUP_ID2} ${GROUP_NAME2} && \
     useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${GROUP_ID1} -G sudo,${GROUP_ID2} -u ${USER_ID} ${USER_NAME} && \
     chown ${USER_ID} -R /home/${USER_NAME} && \
     echo -e "${USER_NAME}\n${USER_NAME}" | passwd ${USER_NAME}

# Set some basic ENV vars for readability
ENV HOME=/home/${USER_NAME}
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda

# WORKDIR instruction sets the directory the following instructions should be run from
WORKDIR ${HOME}

# [optional and recommended] Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p ${CONDA_PREFIX}
RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash

# [optional] Github user configuration (necessary for accessing private repos)
RUN --mount=type=secret,id=dot_env source /run/secrets/dot_env && \
    git config --global user.name ${GITHUB_NAME}
RUN --mount=type=secret,id=dot_env source /run/secrets/dot_env && \
    git config --global user.email ${GITHUB_EMAIL}
# RUN git config --global pull.rebase false

# Prepare the NFS mount
RUN mkdir /mnt/dlabdata1
RUN mkdir /mnt/scratch

# Avoid ascii errors when reading files in Python
# https://stackoverflow.com/a/60084243/12234753
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'


##############################################
#
#        project specific
#
##############################################

# Create conda environment and install requirements
COPY docker/requirements.txt .
ENV ENV_NAME=${PROJECT_NAME}
RUN ${CONDA} create -y -n ${ENV_NAME} python=3.9
RUN ${CONDA} run --name ${ENV_NAME} pip install -r requirements.txt

# [Optional] Login to wandb
RUN --mount=type=secret,id=dot_env source /run/secrets/dot_env && \
    ${CONDA} run -n ${ENV_NAME} wandb login ${WANDB_API_KEY}

# Login to HF
RUN --mount=type=secret,id=dot_env source /run/secrets/dot_env && \
     ${CONDA} run -n ${ENV_NAME} python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('${HF_API_KEY}')"
#################################################
#
#          GF
#
##################################################

RUN wget https://github.com/GrammaticalFramework/gf-core/releases/download/3.11/gf-3.11-ubuntu-20.04.deb
RUN apt-get install ./gf-3.11-ubuntu-20.04.deb


# Change ownership of the whole /home/USER forlder, so that the files created by root can be accesible by world user, such as git cloned repo etc.
# By default, they are owned by root.
RUN --mount=type=secret,id=dot_env source /run/secrets/dot_env && \
    chown ${USER_ID} -R /home/${USER_NAME}


# fix problem running a Haskell app on an Ubuntu docker image.
# https://github.com/snoyberg/http-client/issues/292
RUN apt-get install netbase
#################################################
#
#          Install zsh
#
##################################################
RUN apt install -y zsh

# N.B. the following commands are run as the user, not as root; Don't move them above the USER command
# Uses "Spaceship" theme with some customization. Uses some bundled plugins and installs some more from github
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

USER ${USER_NAME}

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY docker/entrypoint.sh /entrypoint.sh

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]
