# Choose a docker template
# This will set what OS, CUDA, and perhaps even packages / python versions
# you can preemptly have. You can find more templates in
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.02-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN echo "Installing Apex on top of ${BASE_IMAGE}"
# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /workspace

#################################################
#
# Don't change the above lines, otherwise you may get error with installing apex
# I'm not sure about the error and don't want to spend time on it, so please don't change it
# unless you know what you are doing
#
#################################################

# use /bin/bash to execute the SHELL command, by default it is /bin/sh
# -c means run the command and -u means fail if the command fails
SHELL ["/bin/bash", "-cu"]

# Setup your user profile with the right group permission to access NFS folder
# For the command that gives the ids and names you would need checkout the .env file
WORKDIR /

# force recompile
RUN echo "20230512" >/dev/null

# install sudo
RUN apt update && \
    apt -y install sudo

# create a new user and add it to the relevant groups, including sudo group
RUN --mount=type=secret,id=dot_env source /run/secrets/dot_env && \
     groupadd -g ${GROUP_ID1} ${GROUP_NAME1} && \
     groupadd -g ${GROUP_ID2} ${GROUP_NAME2} && \
     useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${GROUP_ID1} -G sudo,${GROUP_ID2} -u ${USER_ID} ${USER_NAME} && \
     chown ${USER_ID} -R /home/${USER_NAME} && \
     echo -e "${USER_NAME}\n${USER_NAME}" | passwd ${USER_NAME}

# Set some basic ENV vars for readability
ENV USER_NAME="geng"
ENV HOME=/home/${USER_NAME}
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda

# the repo name on github
#ENV REPO=llama
#ENV REPO_DIR=${HOME}/${REPO}

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
RUN echo "20230308" >/dev/null


# Create conda environment and install requirements
COPY requirements.txt .
ENV ENV_NAME="gcd"
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
