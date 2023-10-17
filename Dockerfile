FROM python:3.8 as base

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
 apt-get install -y \
 python3-pip \
 python-is-python3 \
 ssh \
 git \
 ffmpeg \
 libsm6 \
 libxext6 \
 libpq-dev \
 wget


# Install miniconda
FROM base as miniconda-setup

ENV MINICONDA_VERSION 4.8.2
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash
RUN bash


# Set up SSH:
FROM miniconda-setup as ssh-setup

ARG SSH_PRIVATE_KEY
RUN echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_ecdsa
RUN chmod 600 /root/.ssh/id_ecdsa
# Format key correctly
RUN sed -i -e "s/-----BEGIN OPENSSH PRIVATE KEY-----/&\n/"\
    -e "s/-----END OPENSSH PRIVATE KEY-----/\n&/"\
    -e "s/\S\{70\}/&\n/g"\
    /root/.ssh/id_ecdsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
# Ensure ssh is setup correctly:
RUN ssh -T git@github.com ; test $? -eq 1 


# Create rogues-venv conda environment
FROM ssh-setup as conda-env-setup

WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml


# Python package setup
FROM conda-env-setup as python-setup

COPY ./ground_data_processing /app/ground_data_processing
COPY ./ddb_tracking /app/ddb_tracking
COPY ./analysis /app/analysis
COPY pyproject.toml /app
RUN conda run -n rogue-venv pip install -e .

ARG VERSION
ENV VERSION=$VERSION

# Don't buffer Python stdout (logs)
ENV PYTHONUNBUFFERED 1
ENV ECS_AVAILABLE_LOGGING_DRIVERS="awslogs"