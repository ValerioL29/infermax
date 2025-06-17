#####################################
# RCP CaaS requirement (Image)
#####################################
# The best practice is to use an image
# with GPU support pre-built by Nvidia.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

FROM nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID} && \
    useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME} && \
    mkdir -p /home/${LDAP_USERNAME} && \
    chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME} && \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential ca-certificates git curl git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#####################################
# DIAS user setup
#####################################
# Create the user and set the working directory
USER ${LDAP_USERNAME}

# Install micromamba for python package management
COPY --from=mambaorg/micromamba:2.3.0-ubuntu24.04 /bin/micromamba /usr/local/bin/micromamba
ENV MAMBA_ROOT_PREFIX=/home/${LDAP_USERNAME}/micromamba

# Copy the code and set the working directory
COPY ./ /home/${LDAP_USERNAME}/infermax
WORKDIR /home/${LDAP_USERNAME}/infermax

# Micromamba initialization
RUN micromamba shell init -s bash -r /home/${LDAP_USERNAME}/micromamba && . /home/${LDAP_USERNAME}/.bashrc && \
    micromamba config append channels conda-forge && \
    micromamba config set channel_priority flexible && \
    micromamba create -n inf python=3.10 -y && \
    micromamba run -n inf pip install --no-cache-dir vllm==0.6.3 && \
    micromamba run -n inf pip install --no-cache-dir -r requirements.txt && \
    micromamba run -n inf pip install --no-cache-dir -r vidur/requirements.txt && \
    micromamba clean -a -y

# Entrypoint
CMD ["/bin/bash"]
